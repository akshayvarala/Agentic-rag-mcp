"""Indexing metadata tracking."""

import json
from pathlib import Path
from typing import Any

from rag.logging import get_logger

log = get_logger("metadata")


class IndexMetadata:
    def __init__(self, metadata_dir: Path):
        self._dir = metadata_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, document_id: str, data: dict[str, Any]) -> None:
        path = self._dir / f"{document_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        log.info("Saved metadata for %s", document_id)

    def load(self, document_id: str) -> dict[str, Any] | None:
        path = self._dir / f"{document_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, document_id: str) -> None:
        path = self._dir / f"{document_id}.json"
        if path.exists():
            path.unlink()
            log.info("Deleted metadata for %s", document_id)

    def list_all(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.json")]
