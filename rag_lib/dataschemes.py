from pydantic import BaseModel, model_validator, Field


class Node(BaseModel):
    id: int
    file_name: str
    parent_id: int | None
    child_ids: list[int] = []
    description: str | None = None
    keywords: list[str] = []
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @model_validator(mode="after")
    def check_page_range(self):
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self

    def get_sparse_text(self):
        return f"{self.file_name}, {self.description or 'N/A'}, {', '.join(self.keywords) or 'N/A'}"

    def get_dense_text(self):
        return f"{self.description or 'N/A'}"


class Chunk(BaseModel):
    file_name: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str

    @model_validator(mode="after")
    def check_page_range(self):
        if self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class DescriptorOutput(BaseModel):
    description: str = Field(
        description=(
            "A concise technical summary (3-5 sentences) optimized for semantic vector search. "
            "For individual pages: extract specific requirements, data, and facts. "
            "For parent nodes: synthesize an overarching theme that aggregates all child nodes. "
            "Avoid introductory phrases and maintain a professional tone."
        ),
    )
    keywords: list[str] = Field(
        description=(
            "A list of 5-10 unique anchor terms in their base form for exact keyword search (BM25). "
            "Must include technical specifics, abbreviations, synonyms, and English equivalents. "
            "Focus on high-entropy tokens that distinguish this content from others."
        ),
    )
