from pathlib import Path
from typing import Optional

from rich.console import Console
from openai import OpenAI
from qdrant_client import QdrantClient

from .config import API_KEY, URL_BASE

console = Console()

lib_path = Path(__file__).parent


class RAGalicClient:
    _instance: Optional["RAGalicClient"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, path: str = str(lib_path / "qdrant_db"), **kwargs):
        if self._initialized:
            return
        self.path = path
        self.kwargs = kwargs
        self._client = None
        self._initialized = True

    def _setup_connection(self):
        """Внутренний метод для реального подключения"""
        if self._client is None:
            self._client = QdrantClient(path=self.path, **self.kwargs)
            self._client.set_model("jinaai/jina-embeddings-v3")
            self._client.set_sparse_model("Qdrant/bm25")
            console.print(
                "Successfully connected to Qdrant.", style="italic bright_black"
            )

    def __enter__(self):
        self._setup_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            console.print("Connection closed.", style="italic bright_black")

    @property
    def client(self):
        if self._client is None:
            self._setup_connection()
        return self._client


class OpenAIClient:
    _instance: Optional["OpenAIClient"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self, api_key: str | None = None, url_base: str | None = None, **kwargs
    ):
        if self._initialized:
            return
        self.api_key = api_key or API_KEY
        self.url_base = url_base or URL_BASE
        self.kwargs = kwargs
        self._client = None
        self._initialized = True

    def _setup_connection(self):
        """Внутренний метод для реального подключения"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.url_base, **self.kwargs
            )
            console.print(
                "Successfully connected to OpenAI.", style="italic bright_black"
            )

    def __enter__(self):
        self._setup_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
            console.print("Connection closed.", style="italic bright_black")

    @property
    def client(self):
        if self._client is None:
            self._setup_connection()
        return self._client


# Глобальный экземпляр (ленивая инициализация)
def get_ragalic_client(**kwargs) -> RAGalicClient:
    client = RAGalicClient(**kwargs)
    client._setup_connection()
    return client


def get_openai_client(**kwargs) -> OpenAIClient:
    client = OpenAIClient(**kwargs)
    client._setup_connection()
    return client
