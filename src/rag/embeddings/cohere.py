"""Cohere embedding provider via API."""

import os
from typing import Any

from rag.embeddings.base import EmbeddingProvider
from rag.errors import EmbeddingError
from rag.logging import get_logger

log = get_logger("embeddings.cohere")

# Model dimensions
COHERE_DIMENSIONS: dict[str, int] = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
    "embed-english-v2.0": 1024,
    "embed-multilingual-v2.0": 1024,
    "embed-english-light-v2.0": 1024,
    "embed-multilingual-light-v2.0": 1024,
}


class CohereEmbedding(EmbeddingProvider):
    """Cohere API embedding provider."""

    def __init__(
        self,
        model_name: str = "embed-english-v3.0",
        api_key: str | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        self._model_name = model_name
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._batch_size = batch_size
        self._client = None
        self._dimension = COHERE_DIMENSIONS.get(model_name)

        if not self._api_key:
            raise EmbeddingError(
                "Cohere API key required. Set COHERE_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def _load_model(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            import cohere

            log.info("Initializing Cohere client for model: %s", self._model_name)
            self._client = cohere.Client(api_key=self._api_key)
            return self._client
        except ImportError:
            raise EmbeddingError(
                "cohere package not installed. Install with: pip install cohere"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize Cohere client: {e}", cause=e)

    def _get_dimension(self) -> int:
        """Get embedding dimension from API response if not known."""
        if self._dimension:
            return self._dimension

        # Make a test call to get dimension
        client = self._load_model()
        try:
            response = client.embed(
                model=self._model_name,
                texts=["test"],
                input_type="search_document",
            )
            self._dimension = len(response.embeddings[0])
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
                response = client.embed(
                    model=self._model_name,
                    texts=batch,
                    input_type="search_document",
                    embedding_types=["float"],
                )
                batch_embeddings = response.embeddings.float
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise EmbeddingError(f"Failed to embed document batch: {e}", cause=e)

        log.info("Embedded %d documents via Cohere", len(texts))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        client = self._load_model()
        try:
            response = client.embed(
                model=self._model_name,
                texts=[query],
                input_type="search_query",
                embedding_types=["float"],
            )
            return response.embeddings.float[0]
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}", cause=e)

    def dimension(self) -> int:
        return self._get_dimension()

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "batch_size": self._batch_size,
            "backend": "cohere",
        }
