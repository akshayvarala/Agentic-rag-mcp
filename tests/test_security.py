"""Tests for security validation."""

import pytest
from pathlib import Path

from rag.mcp.security import validate_path, validate_file_size, validate_query
from rag.errors import SecurityError


class TestValidatePath:
    def test_empty_path(self, tmp_path):
        with pytest.raises(SecurityError, match="Path cannot be empty"):
            validate_path("", tmp_path)

    def test_null_bytes(self, tmp_path):
        with pytest.raises(SecurityError, match="null bytes"):
            validate_path("file\x00.txt", tmp_path)

    def test_absolute_path(self, tmp_path):
        with pytest.raises(SecurityError, match="Absolute paths not allowed"):
            validate_path("/etc/passwd", tmp_path)

    def test_path_traversal(self, tmp_path):
        with pytest.raises(SecurityError, match="Path traversal not allowed"):
            validate_path("../../../etc/passwd", tmp_path)

    def test_valid_relative_path(self, tmp_path):
        (tmp_path / "test.txt").write_text("test")
        result = validate_path("test.txt", tmp_path)
        assert result == tmp_path / "test.txt"

    def test_nested_valid_path(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "test.txt").write_text("test")
        result = validate_path("subdir/test.txt", tmp_path)
        assert result == tmp_path / "subdir" / "test.txt"


class TestValidateFileSize:
    def test_file_within_limit(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("small")
        validate_file_size(f, max_size_mb=1)

    def test_file_exceeds_limit(self, tmp_path):
        f = tmp_path / "large.txt"
        f.write_text("x" * (2 * 1024 * 1024))
        with pytest.raises(SecurityError, match="File too large"):
            validate_file_size(f, max_size_mb=1)


class TestValidateQuery:
    def test_empty_query(self):
        with pytest.raises(SecurityError, match="Query cannot be empty"):
            validate_query("")

    def test_whitespace_query(self):
        with pytest.raises(SecurityError, match="Query cannot be empty"):
            validate_query("   ")

    def test_long_query(self):
        with pytest.raises(SecurityError, match="Query too long"):
            validate_query("x" * 10001)

    def test_valid_query(self):
        validate_query("This is a valid query")
