"""Sentence-Transformers embedding provider as fallback for llama-cpp-python."""

from typing import Any

from rag.embeddings.base import EmbeddingProvider
from rag.errors import EmbeddingError
from rag.logging import get_logger

log = get_logger("embeddings.st_local")


class STLocalEmbedding(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dimension = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            log.info("Loading sentence-transformers model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            log.info("Model loaded, dimension=%d", self._dimension)
            return self._model
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load sentence-transformers model: {e}", cause=e)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._load_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            log.info("Embedded %d documents", len(texts))
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to embed documents: {e}", cause=e)

    def embed_query(self, query: str) -> list[float]:
        model = self._load_model()
        try:
            embedding = model.encode(
                [query],
                normalize_embeddings=True,
            )
            return embedding[0].tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}", cause=e)

    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "dimension": self._dimension,
            "batch_size": self._batch_size,
            "backend": "sentence-transformers",
        }
