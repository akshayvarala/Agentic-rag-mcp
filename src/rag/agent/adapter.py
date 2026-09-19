"""Thin adapter for agentic environment integration."""

from pathlib import Path
from typing import Any

from rag.logging import get_logger
from rag.mcp.tools import RAGTools
from rag.pipeline import RAGPipeline

log = get_logger("agent.adapter")


class RAGAgent:
    def __init__(self, pipeline: RAGPipeline, allowed_input_dir: Path):
        self._pipeline = pipeline
        self._tools = RAGTools(pipeline, allowed_input_dir)
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._pipeline.initialize()
        self._initialized = True
        log.info("RAGAgent initialized")

    def search(self, query: str, top_k: int = 10, **filters: Any) -> dict[str, Any]:
        self.initialize()
        return self._tools.search(query, top_k=top_k, **filters)

    def ingest(self, relative_path: str) -> dict[str, Any]:
        self.initialize()
        return self._tools.ingest(relative_path)

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        self.initialize()
        return self._tools.get_chunk(chunk_id)

    def status(self) -> dict[str, Any]:
        return self._tools.status()

    def delete(self, document_id: str) -> dict[str, Any]:
        self.initialize()
        return self._tools.delete(document_id)

    def health_check(self) -> bool:
        try:
            status = self.status()
            return status.get("initialized", False)
        except Exception:
            return False

    def shutdown(self) -> None:
        log.info("RAGAgent shutting down")
        self._initialized = False
