"""MCP tool definitions for the RAG system."""

from pathlib import Path
from typing import Any

from rag.mcp.security import validate_path, validate_file_size, validate_query
from rag.pipeline import RAGPipeline
from rag.logging import get_logger

log = get_logger("mcp.tools")


class RAGTools:
    def __init__(self, pipeline: RAGPipeline, allowed_input_dir: Path):
        self._pipeline = pipeline
        self._allowed_input_dir = allowed_input_dir

    def ingest(self, relative_path: str) -> dict[str, Any]:
        log.info("MCP tool: ingest(%s)", relative_path)
        resolved = validate_path(relative_path, self._allowed_input_dir)
        validate_file_size(resolved)
        return self._pipeline.ingest(resolved)

    def search(self, query: str, top_k: int = 10, document_id: str | None = None, file_type: str | None = None, source: str | None = None) -> dict[str, Any]:
        log.info("MCP tool: search(%s, top_k=%d)", query[:80], top_k)
        validate_query(query)

        filters = {}
        if document_id:
            filters["source_document"] = document_id
        if file_type:
            filters["file_type"] = file_type
        if source:
            filters["source_file"] = source

        return self._pipeline.search(query, top_k=top_k, filters=filters or None)

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        log.info("MCP tool: get_chunk(%s)", chunk_id)
        if not chunk_id or not chunk_id.strip():
            return None
        return self._pipeline.get_chunk(chunk_id)

    def status(self) -> dict[str, Any]:
        log.info("MCP tool: status()")
        return self._pipeline.status()

    def delete(self, document_id: str) -> dict[str, Any]:
        log.info("MCP tool: delete(%s)", document_id)
        if not document_id or not document_id.strip():
            return {"error": "document_id cannot be empty"}
        return self._pipeline.delete(document_id)
