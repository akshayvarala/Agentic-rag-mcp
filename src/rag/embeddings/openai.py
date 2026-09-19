"""OpenAI embedding provider via API."""

import os
from typing import Any

from rag.embeddings.base import EmbeddingProvider
from rag.errors import EmbeddingError
from rag.logging import get_logger

log = get_logger("embeddings.openai")

# Model dimensions
OPENAI_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI API embedding provider."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._batch_size = batch_size
        self._client = None
        self._dimension = OPENAI_DIMENSIONS.get(model_name)

        if not self._api_key:
            raise EmbeddingError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def _load_model(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI

            log.info("Initializing OpenAI client for model: %s", self._model_name)
            self._client = OpenAI(api_key=self._api_key)
            return self._client
        except ImportError:
            raise EmbeddingError(
                "openai package not installed. Install with: pip install openai"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI client: {e}", cause=e)

    def _get_dimension(self) -> int:
        """Get embedding dimension from API response if not known."""
        if self._dimension:
            return self._dimension

        # Make a test call to get dimension
        client = self._load_model()
        try:
            response = client.embeddings.create(
                model=self._model_name,
                input=["test"],
            )
            self._dimension = len(response.data[0].embedding)
            return self._dimension
        except Exception as e:
            raise EmbeddingError(f"Failed to determine embedding dimension: {e}", cause=e)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._load_model()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = client.embeddings.create(
                    model=self._model_name,
                    input=batch,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise EmbeddingError(f"Failed to embed document batch: {e}", cause=e)

        log.info("Embedded %d documents via OpenAI", len(texts))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        client = self._load_model()
        try:
            response = client.embeddings.create(
                model=self._model_name,
                input=[query],
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}", cause=e)

    def dimension(self) -> int:
        return self._get_dimension()

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "batch_size": self._batch_size,
            "backend": "openai",
        }
