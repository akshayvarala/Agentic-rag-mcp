"""Path sandboxing and input validation for MCP server."""

from pathlib import Path

from rag.errors import SecurityError
from rag.logging import get_logger

log = get_logger("mcp.security")


def validate_path(file_path: str, allowed_dir: Path) -> Path:
    if not file_path or not file_path.strip():
        raise SecurityError("Path cannot be empty")

    if "\x00" in file_path:
        raise SecurityError("Path contains null bytes")

    path = Path(file_path)

    if path.is_absolute():
        if not path.exists():
            raise SecurityError(f"File not found: {file_path}")
        if not path.is_file():
            raise SecurityError(f"Not a file: {file_path}")
        return path.resolve()

    if ".." in file_path.split("/") or ".." in file_path.split("\\"):
        raise SecurityError(f"Path traversal not allowed: {file_path}")

    resolved = (allowed_dir / file_path).resolve()
    return resolved


def validate_file_size(path: Path, max_size_mb: int = 100) -> None:
    if path.exists() and path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise SecurityError(f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")


def validate_query(query: str, max_length: int = 10000) -> None:
    if not query or not query.strip():
        raise SecurityError("Query cannot be empty")
    if len(query) > max_length:
        raise SecurityError(f"Query too long: {len(query)} chars (max: {max_length})")
