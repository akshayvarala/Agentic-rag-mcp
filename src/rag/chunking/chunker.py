"""Text chunking with metadata preservation."""

import uuid
from dataclasses import dataclass, field
from typing import Any

from rag.errors import ChunkingError
from rag.logging import get_logger

log = get_logger("chunking")


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    text: str = ""
    index: int = 0
    start_char: int = 0
    end_char: int = 0
    source_document: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        if chunk_size <= 0:
            raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ChunkingError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ChunkingError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, source_document: str = "", metadata: dict[str, Any] | None = None) -> list[Chunk]:
        if not text.strip():
            return []

        meta = metadata or {}
        chunks: list[Chunk] = []

        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                break_point = self._find_break_point(text, start, end)
                if break_point > start:
                    end = break_point

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start,
                    end_char=end,
                    source_document=source_document,
                    metadata=meta,
                ))
                idx += 1

            next_start = end - self.chunk_overlap
            if next_start <= start:
                start = end
            else:
                start = next_start
            if start >= len(text):
                break

        log.info("Chunked %s into %d chunks (size=%d, overlap=%d)", source_document or "text", len(chunks), self.chunk_size, self.chunk_overlap)
        return chunks

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        search_range = text[start:end]

        for sep in ["\n\n", "\n", ". ", " "]:
            last_sep = search_range.rfind(sep)
            if last_sep > self.chunk_size // 2:
                return start + last_sep + len(sep)

        return end
