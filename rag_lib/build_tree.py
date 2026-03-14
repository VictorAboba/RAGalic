from pathlib import Path
import json
import uuid
import math

from rich.console import Console
from qdrant_client.models import PointStruct, Document

from .clients import RAGalicClient
from .chunking import chunk_document
from .utils import llm_call
from .dataschemes import DescriptorOutput, Node

DESCRIPTOR_SYSTEM_PROMPT = """# Instruction

## Role
You are an expert in analyzing technical and regulatory documentation. Your task is to generate metadata for a document tree structure.

## Input Context
You will receive one of the following as input:
1. **Raw Page Content**: A fragment of text from a single document page.
2. **Child Metadata**: A collection of descriptions and keywords from sub-nodes (chapters or pages) that need to be aggregated.

## Task Logic
* **Direct Processing**: If raw text is provided, extract its core technical essence and specific details.
* **Hierarchical Aggregation**: If child descriptions are provided, synthesize them into a higher-level parent summary. Do not simply list the sub-nodes; identify the overarching theme that unites them.

## Rules
* **Description**: Formulate a brief technical summary (3-5 sentences).
* For a page: Define specific subjects, requirements, or data presented.
* For a parent node: Create a generalizing definition that covers the scope of all child nodes.
* Use professional terminology, avoid introductory phrases.

* **Keywords**: Highlight 5-10 unique anchor terms in their base form.
* Include: specific details, abbreviations, synonyms, and English equivalents.
* For parent nodes, include terms that define the entire section's domain.

## Constraints
1. Do not duplicate the fragment text verbatim.
2. If the input is empty or contains only artifacts, provide a general description based on the inferred document context.
3. Ensure the description is optimized for **semantic (vector) search** and keywords for **keyword (BM25) search**.

## Response Format
Return ONLY a valid JSON object. Do not include markdown blocks like ` ` `json. Do not use double quotes inside string values unless escaped.
* `description`: Technical summary (string).
* `keywords`: List of terms (array of strings)."""

PATH_TO_PARSED_DOCS = Path(__file__).parent / "database" / "parsed_files"

PATH_TO_PARSED_DOCS.mkdir(511, parents=True, exist_ok=True)

console = Console()


def build_tree(path: Path, width: int = 3, batch_size: int = 5):
    name = path.name
    console.print(f"Building tree for: {name}", style="bold black on white")
    chunks = chunk_document(path)

    if len(chunks) == 0:
        console.print(f"No valid chunks found for {name}.", style="bold red")
        raise ValueError(f"No valid chunks found for {name}.")

    with open(
        PATH_TO_PARSED_DOCS / f"{path.with_suffix('.json').name}", "w", encoding="utf-8"
    ) as f:
        json.dump([""] + chunks, f, ensure_ascii=False, indent=4)

    with RAGalicClient() as client:
        if not client.client.collection_exists("ragalic"):
            client.client.create_collection(
                collection_name="ragalic",
                vectors_config={
                    "dense": list(client.client.get_fastembed_vector_params().items())[
                        0
                    ][1]
                },  # Конфиг для Dense
                sparse_vectors_config={
                    "sparse": list(
                        client.client.get_fastembed_sparse_vector_params().items()
                    )[0][1]
                },  # Конфиг для Sparse
            )
        cur_id = client.client.count("ragalic").count

    nodes: list[Node] = []
    for i, chunk in enumerate(chunks, start=1):
        console.print(
            f"--- RUNNING LEAVES [ {i}/{len(chunks)} ] [ FILE: {name} ] ---",
            style="bold violet",
        )
        messages = [
            {"role": "system", "content": DESCRIPTOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Raw Page Content:\n{chunk}"},
        ]
        retries = 0
        while retries < 3:
            output_str = "Not initialized!"
            try:
                output_str, reasoning = llm_call(messages, DescriptorOutput)
                console.print("*" * 20, "MODEL REASONING", style="italic bold cyan")
                console.print(reasoning, style="italic blue")
                console.print("*" * 20, "MODEL RESPONSE", style="italic bold yellow")
                console.print_json(output_str)
                output = DescriptorOutput.model_validate_json(output_str)
                break
            except Exception as e:
                console.print(
                    f"Model response: {output_str}",
                    style="italic bright_black",
                )
                console.print(
                    f"Error during LLM call (for desciption): {e}. Retrying ({retries + 1}/3)...",
                    style="bold red",
                )
                retries += 1
        else:
            console.print(
                "Failed to get a valid response after 3 attempts. Using default values.",
                style="bold underline red on white",
            )
            output = DescriptorOutput(
                description="No description available.", keywords=[]
            )
        node = Node(
            id=cur_id,
            file_name=name,
            parent_id=None,
            child_ids=[],
            description=output.description,
            keywords=output.keywords,
            page_start=i,
            page_end=i,
        )
        nodes.append(node)
        cur_id += 1

    def create_parent_node_and_update_children(child_nodes: list[Node], cur_id) -> Node:
        combined_metadata = "\n\n".join(
            [
                f"Child #{i}\nDescription: {str(node.description)}\nKeywords: {', '.join(node.keywords)}"
                for i, node in enumerate(child_nodes, 1)
            ]
        )
        messages = [
            {"role": "system", "content": DESCRIPTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Child Metadata:\n{combined_metadata}",
            },
        ]
        retries = 0
        while retries < 3:
            output_str = "Not initialized!"
            try:
                output_str, reasoning = llm_call(messages, DescriptorOutput)
                console.print("*" * 20, "MODEL REASONING", style="italic bold cyan")
                console.print(reasoning, style="italic blue")
                console.print("*" * 20, "MODEL RESPONSE", style="italic bold yellow")
                console.print_json(output_str)
                output = DescriptorOutput.model_validate_json(output_str)
                break
            except Exception as e:
                console.print(
                    f"Model response: {output_str}",
                    style="italic bright_black",
                )
                console.print(
                    f"Error during LLM call (for parent node description): {e}. Retrying ({retries + 1}/3)...",
                    style="bold red",
                )
                retries += 1
        else:
            console.print(
                "Failed to get a valid response after 3 attempts. Using default values for parent node.",
                style="bold underline red on white",
            )
            output = DescriptorOutput(
                description="No description available.", keywords=[]
            )

        start_page = min(child_nodes, key=lambda n: n.page_start).page_start
        end_page = max(child_nodes, key=lambda n: n.page_end).page_end
        all_child_keywords = list(
            set(kw for node in child_nodes for kw in node.keywords)
        )

        parent_node = Node(
            id=cur_id,
            file_name=name,
            parent_id=None,
            child_ids=[node.id for node in child_nodes],
            description=output.description,
            keywords=all_child_keywords,
            page_start=start_page,
            page_end=end_page,
        )

        for node in child_nodes:
            node.parent_id = parent_node.id

        return parent_node

    # Create parent nodes for every `width` child nodes
    nodes_to_process = nodes.copy()
    while len(nodes_to_process) > 1:
        will_have_new_nodes = math.ceil(len(nodes_to_process) / width)
        console.print(
            f"--- CREATING NODES [ {len(nodes_to_process)} -> {will_have_new_nodes} ] ---",
            style="bold violet",
        )
        new_nodes = []
        for i in range(0, len(nodes_to_process), width):
            child_nodes = nodes_to_process[i : i + width]
            if len(child_nodes) == 1:
                console.print(
                    f"--- NEW SINGLE NODE [ {i // width + 1}/{will_have_new_nodes} ] [ FILE: {name}] ---",
                    style="dark_violet",
                )
                new_nodes.append(child_nodes[0])  # No need to create a parent node
            else:
                console.print(
                    f"--- NEW COMPOUND NODE [ {len(child_nodes)} -> 1 ] [ {i // width + 1}/{will_have_new_nodes} ] [ FILE: {name}] ---",
                    style="dark_violet",
                )
                parent_node = create_parent_node_and_update_children(
                    child_nodes, cur_id
                )
                nodes.append(parent_node)
                new_nodes.append(parent_node)
                cur_id += 1
        nodes_to_process = new_nodes

    # Root should have parent_id = -1
    for node in nodes:
        if node.parent_id is None:
            node.parent_id = -1

    with RAGalicClient() as client:
        num_batches = (len(nodes) + batch_size - 1) // batch_size
        for i in range(0, len(nodes), batch_size):
            console.print(
                f"Upserting batch {i // batch_size + 1}/{num_batches} [{name}]...",
                style="italic black on yellow",
            )
            batch = nodes[i : i + batch_size]
            points = [
                PointStruct(
                    id=uuid.uuid5(uuid.NAMESPACE_DNS, str(node.id)),
                    vector={
                        "sparse": Document(
                            text=node.get_sparse_text(),
                            model=client.client.sparse_embedding_model_name,  # type: ignore
                        ),
                        "dense": Document(
                            text=node.get_dense_text(),
                            model=client.client.embedding_model_name,  # type: ignore
                        ),
                    },
                    payload=node.model_dump(),
                )
                for node in batch
            ]
            client.client.upsert(
                collection_name="ragalic",
                points=points,
                wait=True,
            )

    console.print(f"Finished building tree for: {name}", style="bold green on white")


if __name__ == "__main__":
    test_files = [
        "/home/victor/pet_projects/RAGalic/rag_lib/database/test_1.pdf",
        "/home/victor/pet_projects/RAGalic/rag_lib/database/test_2.pdf",
        "/home/victor/pet_projects/RAGalic/rag_lib/database/test_3.pdf",
    ]
    for file in test_files:
        build_tree(Path(file))
