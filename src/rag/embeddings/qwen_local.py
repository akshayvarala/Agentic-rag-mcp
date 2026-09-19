"""Qwen GGUF embedding provider via llama-cpp-python."""

from pathlib import Path
from typing import Any

from rag.embeddings.base import EmbeddingProvider
from rag.errors import EmbeddingError
from rag.logging import get_logger

log = get_logger("embeddings.qwen_local")

DOCUMENT_INSTRUCTION = "Represent this document for retrieval: "
QUERY_INSTRUCTION = "Represent this question for searching relevant documents: "


class QwenLocalEmbedding(EmbeddingProvider):
    def __init__(
        self,
        model_path: Path,
        dimension: int = 1024,
        batch_size: int = 32,
    ):
        self._model_path = model_path
        self._dimension = dimension
        self._batch_size = batch_size
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if not self._model_path.exists():
            raise EmbeddingError(f"Embedding model not found: {self._model_path}")

        try:
            from llama_cpp import Llama

            log.info("Loading embedding model: %s", self._model_path)
            self._model = Llama(
                model_path=str(self._model_path),
                embedding=True,
                verbose=False,
            )
            log.info("Embedding model loaded successfully")
            return self._model
        except ImportError:
            raise EmbeddingError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}", cause=e)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model = self._load_model()
        prefixed = [DOCUMENT_INSTRUCTION + t for t in texts]

        all_embeddings: list[list[float]] = []
        for i in range(0, len(prefixed), self._batch_size):
            batch = prefixed[i : i + self._batch_size]
            try:
                result = model.create_embedding(batch)
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                raise EmbeddingError(f"Failed to embed document batch: {e}", cause=e)

        log.info("Embedded %d documents", len(texts))
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        model = self._load_model()
        prefixed = QUERY_INSTRUCTION + query

        try:
            result = model.create_embedding([prefixed])
            return result["data"][0]["embedding"]
        except Exception as e:
            raise EmbeddingError(f"Failed to embed query: {e}", cause=e)

    def dimension(self) -> int:
        return self._dimension

    def model_info(self) -> dict[str, Any]:
        return {
            "model_path": str(self._model_path),
            "dimension": self._dimension,
            "batch_size": self._batch_size,
        }
