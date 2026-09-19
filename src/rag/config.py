"""Pydantic settings loaded from environment / .env file."""

import os
import platform
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


def _get_default_data_dir() -> Path:
    """Get platform-appropriate data directory."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "rag-mcp"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding provider settings
    embedding_provider: str = Field(
        default="auto",
        description="Embedding provider: auto, st, gguf, openai, cohere",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name (HuggingFace name for st, model ID for openai/cohere)",
    )
    embedding_model_path: Path | None = Field(
        default=None,
        description="Local GGUF model path (gguf provider only)",
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")
    embedding_batch_size: int = Field(default=32, description="Batch size for document embedding")

    # LLM settings (optional - standalone CLI only)
    llama_cpp_model_path: Path | None = Field(
        default=None,
        description="Path to GGUF LLM model (standalone CLI only)",
    )
    llama_cpp_mmproj_path: Path | None = Field(
        default=None,
        description="Path to multimodal projection GGUF",
    )
    llama_cpp_n_ctx: int = Field(default=4096, description="LLM context size")
    llama_cpp_n_gpu_layers: int = Field(default=-1, description="GPU layers (-1 = all)")
    llama_cpp_temperature: float = Field(default=0.7, description="LLM sampling temperature")

    # Vector store settings
    qdrant_path: Path = Field(
        default_factory=lambda: _get_default_data_dir() / "qdrant",
        description="Qdrant persistence path",
    )
    qdrant_collection: str = Field(default="rag_documents", description="Qdrant collection name")

    # Pipeline settings
    data_dir: Path = Field(
        default_factory=_get_default_data_dir,
        description="Root data directory",
    )
    allowed_input_dir: Path = Field(
        default_factory=lambda: _get_default_data_dir() / "original",
        description="Sandboxed input directory",
    )
    chunk_size: int = Field(default=1024, description="Text chunk size in characters")
    chunk_overlap: int = Field(default=128, description="Overlap between chunks")

    # MCP server settings
    mcp_host: str = Field(default="127.0.0.1", description="MCP server host")
    mcp_port: int = Field(default=3000, description="MCP server port")
    log_level: str = Field(default="INFO", description="Logging level")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
