"""Auto-download embedding models from HuggingFace."""

from pathlib import Path
from typing import Any

from rag.logging import get_logger

log = get_logger("model_downloader")

# Default model cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rag-mcp" / "models"

# Models that can be auto-downloaded
AVAILABLE_MODELS: dict[str, dict[str, Any]] = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "description": "Fast, lightweight (80MB). Good default.",
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "description": "Better quality, slower (420MB).",
    },
    "BAAI/bge-small-en-v1.5": {
        "dimension": 384,
        "description": "BAAI embedding, good retrieval quality.",
    },
    "BAAI/bge-base-en-v1.5": {
        "dimension": 768,
        "description": "BAAI embedding, balanced quality/speed.",
    },
    "BAAI/bge-large-en-v1.5": {
        "dimension": 1024,
        "description": "BAAI embedding, highest quality.",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimension": 384,
        "description": "Sentence-Transformers default.",
    },
}


def get_model_cache_dir() -> Path:
    """Get the model cache directory."""
    cache_dir = Path.home() / ".cache" / "rag-mcp" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_model_cached(model_name: str) -> bool:
    """Check if a model is already cached locally."""
    try:
        from sentence_transformers import SentenceTransformer

        cache_dir = get_model_cache_dir()
        model_path = cache_dir / model_name.replace("/", "_")
        return model_path.exists()
    except Exception:
        return False


def download_model(
    model_name: str,
    cache_dir: Path | None = None,
    show_progress: bool = True,
) -> Any:
    """
    Download a sentence-transformers model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path.
        cache_dir: Directory to cache models. Defaults to ~/.cache/rag-mcp/models/
        show_progress: Show download progress bar.

    Returns:
        Loaded SentenceTransformer model.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        )

    if cache_dir is None:
        cache_dir = get_model_cache_dir()

    log.info("Loading model: %s (cache: %s)", model_name, cache_dir)

    # Sentence Transformers handles caching automatically via HuggingFace hub
    model = SentenceTransformer(model_name)
    log.info("Model loaded successfully: %s", model_name)
    return model


def get_model_info(model_name: str) -> dict[str, Any] | None:
    """Get information about a model."""
    if model_name in AVAILABLE_MODELS:
        info = AVAILABLE_MODELS[model_name].copy()
        info["name"] = model_name
        info["cached"] = is_model_cached(model_name)
        return info
    return None


def list_available_models() -> list[dict[str, Any]]:
    """List all available models with their info."""
    models = []
    for name, info in AVAILABLE_MODELS.items():
        model_info = info.copy()
        model_info["name"] = name
        model_info["cached"] = is_model_cached(name)
        models.append(model_info)
    return models
