from pathlib import Path
import json
from typing import Literal

from rich.console import Console
from qdrant_client.models import (
    Filter,
    FieldCondition,
    ScoredPoint,
    Prefetch,
    Document,
    MatchValue,
    MatchAny,
    FusionQuery,
    Fusion,
)
import numpy as np

from .clients import RAGalicClient
from .dataschemes import Chunk

console = Console()

BEAM_SEARCH_METHODS = Literal["fixed", "adaptive_with_knee"]


def prepare_chunks(points: list[ScoredPoint]) -> list[Chunk]:
    path_to_parsed_files = Path(__file__).parent / "database" / "parsed_files"

    chunks = []
    for point in points:
        name: str = point.payload["file_name"]
        path_to_parsed_file = (path_to_parsed_files / name).with_suffix(".json")
        page_start = point.payload["page_start"]
        page_end = point.payload["page_end"]
        with open(path_to_parsed_file, "r", encoding="utf-8") as file:
            pages_content = json.load(file)
            chunk_lines = [f"FILE NAME: {name}"]
            for i in range(page_start, page_end + 1):
                chunk_lines.append(f"<-- PAGE {i} -->")
                chunk_lines.append(pages_content[i])
        chunk_content = "\n".join(chunk_lines)
        chunk = Chunk(
            file_name=name, page_start=page_start, page_end=page_end, text=chunk_content
        )
        chunks.append(chunk)

    return chunks


def find_roots(query: str, num_to_find: int = 3) -> list[ScoredPoint]:
    console.print(f"Finding roots for query: {query[:20]}...", style="bold violet")
    with RAGalicClient() as client:
        dense_model: str = client.client.embedding_model_name  # type: ignore
        sparse_model: str = client.client.sparse_embedding_model_name  # type: ignore
        root_filter = Filter(
            must=[
                FieldCondition(key="parent_id", match=MatchValue(value=-1)),
            ]
        )
        points = client.client.query_points(  # type: ignore
            collection_name="ragalic",
            prefetch=[
                Prefetch(
                    query=Document(text=query, model=dense_model),
                    using="dense",
                    limit=num_to_find * 3,
                    filter=root_filter,
                ),
                Prefetch(
                    query=Document(text=query, model=sparse_model),
                    using="sparse",
                    limit=num_to_find * 3,
                    filter=root_filter,
                ),
            ],
            query_filter=root_filter,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=num_to_find,
        ).points
    names = [f"{point.payload['file_name']}" for point in points]
    console.print(f"Found roots of files: {names}", style="italic purple")
    return points


##################
# BRANCH SEARCH
##################


def parent_vs_children(query: str, parent: ScoredPoint) -> list[ScoredPoint]:
    file_name = parent.payload["file_name"]
    parent_id = parent.payload["id"]
    child_ids = parent.payload["child_ids"]
    all_ids = [parent_id] + child_ids

    if len(child_ids) == 0:
        console.print(
            f"Parent with id {parent_id} is leave!", style="bold underline violet"
        )
        return []

    console.print()
    console.print("#" * 20, style="red")
    console.print(
        f"Children (ids: {child_ids}) [underline]VS[/underline] Parent (id: {parent_id}) [ FILE: {file_name} ]\nFor query: {query[:20]}...",
        style="bold violet",
    )

    with RAGalicClient() as client:
        dense_model: str = client.client.embedding_model_name  # type: ignore
        sparse_model: str = client.client.sparse_embedding_model_name  # type: ignore
        children_and_parent_filter = Filter(
            must=FieldCondition(key="id", match=MatchAny(any=all_ids))
        )
        sorted_points = client.client.query_points(
            collection_name="ragalic",
            prefetch=[
                Prefetch(
                    query=Document(text=query, model=dense_model),
                    using="dense",
                    limit=len(all_ids),
                    filter=children_and_parent_filter,
                ),
                Prefetch(
                    query=Document(text=query, model=sparse_model),
                    using="sparse",
                    limit=len(all_ids),
                    filter=children_and_parent_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=children_and_parent_filter,
            limit=len(all_ids),
        ).points

    children_better_then_parent = []
    for point in sorted_points:
        if point.payload["id"] == parent_id:
            break
        children_better_then_parent.append(point)

    ids = [child.payload["id"] for child in children_better_then_parent]
    console.print(
        f"Child ids which is better than parent: {ids if ids else 'N/A'}",
        style="italic purple",
    )
    console.print("#" * 20, style="red")
    console.print()

    return children_better_then_parent


def branch_search(query: str, num_roots: int = 3) -> list[Chunk]:
    roots = find_roots(query=query, num_to_find=num_roots)

    final_points = []
    points_to_process = roots
    while len(points_to_process) > 0:
        new_point_to_process = []
        for point in points_to_process:
            new_points = parent_vs_children(query=query, parent=point)
            if len(new_points) == 0:
                console.print(
                    f"--- NEW FINAL POINT (id: {point.payload['id']} | file: {point.payload['file_name']} | pages: {point.payload['page_start']} - {point.payload['page_end']}) ---",
                    style="bold green",
                )
                final_points.append(point)
            else:
                new_point_to_process.extend(new_points)
        points_to_process = new_point_to_process

    console.print(
        f"--- WAS FOUND {len(final_points)} POINTS FOR QUERY: '{query[:20]}...' ---",
        style="bold green on white",
    )

    return prepare_chunks(final_points)


##################
# BEAM SEARCH
##################


def check_ids(old_ids: list, new_ids: list):
    return set(old_ids) == set(new_ids)


def cut_knee(points: list[ScoredPoint]) -> list[ScoredPoint]:
    """
    Отсекает 'хвост' результатов, находя точку максимального изгиба (колено)
    на графике RRF-скоров.
    """
    n_points = len(points)

    if n_points <= 2:
        return points

    scores = np.array([p.score for p in points])
    x = np.arange(n_points)
    y = scores

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    points_vec = np.vstack([x - p1[0], y - p1[1]]).T

    scalar_product = (
        points_vec[:, 0] * line_vec_norm[1] - points_vec[:, 1] * line_vec_norm[0]
    )
    distances = np.abs(scalar_product)

    knee_idx = np.argmax(distances)

    return points[: knee_idx + 1]


def parents_vs_children(
    query: str,
    parents: list[ScoredPoint],
    width: int = 3,
    search_method: BEAM_SEARCH_METHODS = "fixed",
) -> list[ScoredPoint]:
    task_meta = {
        "file_names": [],
        "parent_ids": [],
        "child_ids": [],
    }

    for point in parents:
        task_meta["file_names"].append(point.payload["file_name"])
        task_meta["parent_ids"].append(point.payload["id"])
        task_meta["child_ids"].extend(point.payload["child_ids"])

    task_meta["file_names"] = list(set(task_meta["file_names"]))
    task_meta["parent_ids"] = list(set(task_meta["parent_ids"]))
    task_meta["child_ids"] = list(set(task_meta["child_ids"]))

    if len(task_meta["child_ids"]) == 0:
        console.print(
            f"Parents with ids ({task_meta['parent_ids']}) is all leaves!",
            style="bold underline violet",
        )
        return parents

    console.print("")
    console.print("#" * 20, style="red")
    console.print(
        f"Running [underline]ALL VS ALL[/underline] for:\nParent IDs: {task_meta['parent_ids']} | Files: {task_meta['file_names']}",
        style="bold violet",
    )
    all_ids = task_meta["parent_ids"] + task_meta["child_ids"]

    with RAGalicClient() as client:
        dense_model: str = client.client.embedding_model_name  # type: ignore
        sparse_model: str = client.client.sparse_embedding_model_name  # type: ignore
        all_vs_all_filter = Filter(
            must=FieldCondition(key="id", match=MatchAny(any=all_ids))
        )
        sorted_points = client.client.query_points(
            collection_name="ragalic",
            prefetch=[
                Prefetch(
                    query=Document(text=query, model=dense_model),
                    using="dense",
                    limit=len(all_ids),
                    filter=all_vs_all_filter,
                ),
                Prefetch(
                    query=Document(text=query, model=sparse_model),
                    using="sparse",
                    limit=len(all_ids),
                    filter=all_vs_all_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=all_vs_all_filter,
            limit=len(all_ids),
        ).points

    children_to_eliminate = []
    parents_to_eliminate = []
    new_top_k = []
    for point in sorted_points:
        p_id = point.payload["id"]

        # Логика исключения (Suppression)
        if p_id in task_meta["child_ids"] and p_id not in children_to_eliminate:
            # Если ребенок лучше родителя — родителю тут не место
            parents_to_eliminate.append(point.payload["parent_id"])
            new_top_k.append(point)

        elif p_id in task_meta["parent_ids"] and p_id not in parents_to_eliminate:
            # Если родитель лучше детей — убираем его детей из пула
            children_to_eliminate.extend(point.payload["child_ids"])
            new_top_k.append(point)

        if search_method == "fixed" and len(new_top_k) >= width:
            break

    if search_method == "adaptive_with_knee":
        console.print(
            f"Applying KNEE method to cut the tail of results. Initial candidates: {len(new_top_k)}",
            style="italic bright_black",
        )
        new_top_k = cut_knee(new_top_k)
        console.print(
            f"Candidates after KNEE cut: {len(new_top_k)}",
            style="italic bright_black",
        )

    new_top_k_meta = [
        f"(ID: {point.payload['id']} | FILE: {point.payload['file_name']} | PAGES: {point.payload['page_start']} - {point.payload['page_end']})"
        for point in new_top_k
    ]
    console.print(
        f"IDs: {task_meta['parent_ids']} | Files: {task_meta['file_names']} [OLD TOP]",
        style="italic purple",
    )
    console.print("|\nY", style="bold red")
    console.print(f"{', '.join(new_top_k_meta)} [NEW TOP]", style="italic purple")
    console.print("#" * 20, style="red")
    console.print("")

    return new_top_k


def beam_search(
    query: str,
    beam_width: int = 3,
    search_method: BEAM_SEARCH_METHODS = "fixed",
    max_num_roots: int = 20,
) -> list[Chunk]:
    if search_method == "fixed":
        console.print(
            f"Running BEAM SEARCH with [underline]FIXED[/underline] width: {beam_width}",
            style="bold cyan",
        )
        old_points = find_roots(query=query, num_to_find=beam_width)
    elif search_method == "adaptive_with_knee":
        console.print(
            f"Running BEAM SEARCH with [underline]ADAPTIVE[/underline] width using KNEE method",
            style="bold cyan",
        )
        with RAGalicClient() as client:
            root_filter = Filter(
                must=[
                    FieldCondition(key="parent_id", match=MatchValue(value=-1)),
                ]
            )
            all_root_points_num = client.client.count(
                collection_name="ragalic", count_filter=root_filter
            ).count
            all_root_points_num = min(all_root_points_num, max_num_roots)
        console.print(
            f"Total root points available: {all_root_points_num}",
            style="italic bright_black",
        )
        old_points = find_roots(query=query, num_to_find=all_root_points_num)
        console.print(
            f"Initial root points retrieved: {len(old_points)}",
            style="italic bright_black",
        )
        old_points = cut_knee(old_points)
        console.print(
            f"Root points after KNEE cut: {len(old_points)}",
            style="italic bright_black",
        )
    new_points = []

    old_ids = [point.payload["id"] for point in old_points]
    new_ids = []
    while not check_ids(old_ids=old_ids, new_ids=new_ids):
        old_ids = [point.payload["id"] for point in old_points]
        new_points = parents_vs_children(
            query=query,
            parents=old_points,
            width=beam_width,
            search_method=search_method,
        )
        old_points = new_points
        new_ids = [point.payload["id"] for point in new_points]

    console.print(
        f"--- WAS FOUND {len(new_points)} POINTS FOR QUERY: '{query[:20]}...' ---",
        style="bold green on white",
    )

    return prepare_chunks(new_points)


if __name__ == "__main__":
    test_query = "Summarize the court's final ruling in case CFI 010/2024."
    branch_search(query=test_query)
    beam_search(query=test_query)
    beam_search(query=test_query, search_method="adaptive_with_knee")
