"""Tests for document converters."""

import tempfile
from pathlib import Path

import pytest

from rag.converters.base import ConvertedDocument, DocumentConverter
from rag.errors import ConversionError


class TestDocumentConverterABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DocumentConverter()

    def test_converted_document_dataclass(self):
        doc = ConvertedDocument(
            original_path=Path("test.pdf"),
            markdown_path=Path("test.md"),
            metadata_path=Path("test.json"),
            assets_dir=Path("assets"),
        )
        assert doc.original_path == Path("test.pdf")
        assert doc.metadata == {}


class TestMarkItDownConverter:
    def test_convert_nonexistent_file(self, tmp_path):
        from rag.converters.markitdown import MarkItDownConverter

        converter = MarkItDownConverter()
        with pytest.raises(ConversionError, match="Input file not found"):
            converter.convert(Path("nonexistent.pdf"), tmp_path)

    def test_convert_text_file(self, tmp_path):
        from rag.converters.markitdown import MarkItDownConverter

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_file = input_dir / "test.txt"
        input_file.write_text("Hello World\nThis is a test document.")

        converter = MarkItDownConverter()
        result = converter.convert(input_file, tmp_path / "output")

        assert result.markdown_path.exists()
        assert result.metadata_path.exists()
        assert result.metadata["source_file"] == "test.txt"
        assert result.metadata["file_type"] == ".txt"
