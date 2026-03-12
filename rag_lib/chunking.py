import os

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.page_chunker import PageChunker
from docling_core.types.doc.document import DoclingDocument

load_dotenv()

picture_description_options = PictureDescriptionApiOptions(
    url="https://api.proxyapi.ru/openrouter/v1/chat/completions",  # type: ignore
    headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"},
    params={"model": "mistralai/mistral-small-3.1-24b-instruct:free"},
    prompt="Describe the image.",
    timeout=60,
    scale=1.0,
)


converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True,
                do_picture_description=True,
                picture_description_options=picture_description_options,
                enable_remote_services=True,
                ocr_batch_size=1,
            )
        )
    }
)

chunker = PageChunker()

if __name__ == "__main__":
    document = converter.convert(
        r"C:\Users\admin\Desktop\RAGalic\database\A systematic review of computer vision-based personal protective equipment compliance in industry practice advancements, challenges and future directions.pdf"
    ).document
    chunks = chunker.chunk(document)
    for chunk in chunks:
        print(chunk.text)
        print("*" * 50)
