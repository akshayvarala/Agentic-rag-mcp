"""MarkItDown-based document converter."""

import json
from pathlib import Path

from rag.converters.base import ConvertedDocument, DocumentConverter
from rag.errors import ConversionError
from rag.logging import get_logger

log = get_logger("converters.markitdown")


class MarkItDownConverter(DocumentConverter):
    def convert(self, input_path: Path, output_dir: Path) -> ConvertedDocument:
        if not input_path.exists():
            raise ConversionError(f"Input file not found: {input_path}")

        output_dir.mkdir(parents=True, exist_ok=True)
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        stem = input_path.stem

        try:
            from markitdown import MarkItDown

            md = MarkItDown()
            result = md.convert(str(input_path))
            markdown_content = result.text_content
        except ImportError:
            raise ConversionError("markitdown package not installed. Install with: pip install markitdown[microsoft]")
        except Exception as e:
            raise ConversionError(f"Failed to convert {input_path.name}: {e}", cause=e)

        metadata = self._extract_metadata(input_path, markdown_content)

        markdown_path = output_dir / f"{stem}.md"
        markdown_path.write_text(markdown_content, encoding="utf-8")

        metadata_path = output_dir / f"{stem}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

        log.info("Converted %s -> %s", input_path.name, markdown_path)

        return ConvertedDocument(
            original_path=input_path,
            markdown_path=markdown_path,
            metadata_path=metadata_path,
            assets_dir=assets_dir,
            metadata=metadata,
        )

    def _extract_metadata(self, input_path: Path, content: str) -> dict:
        headings = []
        for line in content.splitlines():
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("#").strip()
                headings.append({"level": level, "text": text})

        return {
            "source_file": input_path.name,
            "file_type": input_path.suffix.lower(),
            "headings": headings,
        }
