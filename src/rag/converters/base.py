"""Base document converter ABC."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConvertedDocument:
    original_path: Path
    markdown_path: Path
    metadata_path: Path
    assets_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentConverter(ABC):
    @abstractmethod
    def convert(self, input_path: Path, output_dir: Path) -> ConvertedDocument:
        """Convert a document to normalized markdown with metadata."""
        ...
