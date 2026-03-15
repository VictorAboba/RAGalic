import re
from pathlib import Path
import warnings

from rich.console import Console
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

warnings.filterwarnings("ignore")

lib_path = Path(__file__).parent

console = Console()

transfer_pattern_left = re.compile(r"(\S)-\s+(\S)")
transfer_pattern_right = re.compile(r"(\S)\s+-(\S)")
multiple_spaces_pattern = re.compile(
    r"[ \t]+",
)
new_line_pattern = re.compile(r"\n +| +\n")

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True,
                ocr_options=EasyOcrOptions(
                    model_storage_directory=str(lib_path / ".." / "./easyocr_models")
                ),
            )
        )
    }
)


def fix_hyphenation(text: str) -> str:
    text = multiple_spaces_pattern.sub(" ", text)
    text = new_line_pattern.sub("\n", text)
    text = transfer_pattern_left.sub(r"\1\2", text)
    text = transfer_pattern_right.sub(r"\1\2", text)
    return text


def chunk_document(path: Path) -> list[str]:
    document = converter.convert(path).document
    num_pages = document.num_pages()
    console.print(f"Document has {num_pages} pages.", style="bold bright_black")
    chunks = []
    for page_num in range(1, num_pages + 1):
        page = document.export_to_markdown(page_no=page_num, indent=2)
        console.print(
            f"Extracted page {page_num} with length {len(page)} characters.",
            style="italic bright_black",
        )
        page = fix_hyphenation(page)
        chunks.extend([page])
    return chunks


if __name__ == "__main__":
    path = Path(
        r"C:\Users\admin\Desktop\RAGalic\database\A systematic review of computer vision-based personal protective equipment compliance in industry practice advancements, challenges and future directions.pdf"
    )
    chunks = chunk_document(path)
    for chunk in chunks:
        print(chunk)
        print("*" * 50)
