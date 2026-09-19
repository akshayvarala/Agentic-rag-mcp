"""Full RAG pipeline orchestration."""

import uuid
from pathlib import Path
from typing import Any

from rag.chunking.chunker import Chunk, TextChunker
from rag.config import Settings, get_settings
from rag.converters.base import ConvertedDocument, DocumentConverter
from rag.embeddings.base import EmbeddingProvider
from rag.errors import RAGError, ConversionError, EmbeddingError, VectorStoreError
from rag.logging import get_logger
from rag.vectorstore.base import VectorStore

log = get_logger("pipeline")


class RAGPipeline:
    def __init__(
        self,
        converter: DocumentConverter,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        settings: Settings | None = None,
    ):
        self._converter = converter
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._settings = settings or get_settings()
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        log.info("Initializing RAG pipeline")
        self._vector_store.create_collection(self._embedding_provider.dimension())
        self._initialized = True
        log.info("Pipeline initialized (dimension=%d)", self._embedding_provider.dimension())

    def ingest(self, input_path: Path, output_dir: Path | None = None) -> dict[str, Any]:
        self.initialize()

        if output_dir is None:
            output_dir = self._settings.data_dir / "normalized" / input_path.stem

        try:
            log.info("Ingesting: %s", input_path)
            converted = self._converter.convert(input_path, output_dir)
            chunks = self._chunker.chunk(
                converted.markdown_path.read_text(encoding="utf-8"),
                source_document=input_path.name,
                metadata=converted.metadata,
            )

            if not chunks:
                return {"document": input_path.name, "chunks": 0, "status": "empty"}

            embeddings = self._embedding_provider.embed_documents([c.text for c in chunks])

            ids = [c.id for c in chunks]
            texts = [c.text for c in chunks]
            metadatas = [
                {
                    "source_document": c.source_document,
                    "chunk_index": c.index,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    **c.metadata,
                }
                for c in chunks
            ]

            self._vector_store.upsert(ids, embeddings, texts, metadatas)

            result = {
                "document": input_path.name,
                "chunks": len(chunks),
                "status": "success",
            }
            log.info("Ingested %s: %d chunks", input_path.name, len(chunks))
            return result

        except (ConversionError, EmbeddingError, VectorStoreError):
            raise
        except Exception as e:
            raise RAGError(f"Ingestion failed for {input_path}: {e}", cause=e)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.initialize()

        from rag.retrieval.retriever import Retriever

        retriever = Retriever(
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
        )

        result = retriever.retrieve(query, top_k=top_k, filters=filters)

        return {
            "query": result.query,
            "total_found": result.total_found,
            "chunks": [
                {
                    "id": c.id,
                    "text": c.text,
                    "score": c.score,
                    "rank": c.rank,
                    "metadata": c.metadata,
                }
                for c in result.chunks
            ],
        }

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        self.initialize()

        client = self._vector_store._get_client()
        uuid_id = self._vector_store._to_uuid(chunk_id)

        try:
            results = client.retrieve(
                collection_name=self._vector_store._collection,
                ids=[uuid_id],
            )
            if not results:
                return None

            r = results[0]
            return {
                "id": chunk_id,
                "text": r.payload.get("text", ""),
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
            }
        except Exception as e:
            log.error("get_chunk failed: %s", e)
            return None

    def delete(self, document_id: str) -> dict[str, Any]:
        self.initialize()

        self._vector_store.delete_by_filter({"source_document": document_id})
        log.info("Deleted document: %s", document_id)
        return {"document": document_id, "status": "deleted"}

    def status(self) -> dict[str, Any]:
        info = self._vector_store.collection_info() if self._initialized else {}
        return {
            "initialized": self._initialized,
            "embedding_model": self._embedding_provider.model_info(),
            "collection": info,
            "chunk_size": self._settings.chunk_size,
            "chunk_overlap": self._settings.chunk_overlap,
        }
