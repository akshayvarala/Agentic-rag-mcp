"""Embedding provider registry - auto-detect or explicit provider selection."""

import importlib
import os
import shutil
from typing import Any

from rag.embeddings.base import EmbeddingProvider
from rag.errors import EmbeddingError
from rag.logging import get_logger

log = get_logger("embeddings.registry")

# Built-in providers: name -> (module_path, class_name)
BUILTIN_PROVIDERS: dict[str, tuple[str, str]] = {
    "st": ("rag.embeddings.st_local", "STLocalEmbedding"),
    "gguf": ("rag.embeddings.qwen_local", "QwenLocalEmbedding"),
    "openai": ("rag.embeddings.openai", "OpenAIEmbedding"),
    "cohere": ("rag.embeddings.cohere", "CohereEmbedding"),
}


def _load_class(module_path: str, class_name: str) -> type[EmbeddingProvider]:
    """Dynamically import a class from a module path."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise EmbeddingError(f"Failed to load provider {module_path}.{class_name}: {e}")


def _load_entry_point(name: str) -> type[EmbeddingProvider] | None:
    """Load a provider from entry points (rag.embeddings group)."""
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        if hasattr(eps, "select"):
            rag_eps = eps.select(group="rag.embeddings")
        else:
            rag_eps = eps.get("rag.embeddings", [])

        for ep in rag_eps:
            if ep.name == name:
                cls = ep.load()
                if isinstance(cls, type) and issubclass(cls, EmbeddingProvider):
                    return cls
    except Exception:
        pass
    return None


def auto_detect_provider() -> str:
    """Auto-detect best available provider based on environment."""
    if os.getenv("OPENAI_API_KEY"):
        log.info("Auto-detected provider: openai (OPENAI_API_KEY set)")
        return "openai"

    if os.getenv("COHERE_API_KEY"):
        log.info("Auto-detected provider: cohere (COHERE_API_KEY set)")
        return "cohere"

    try:
        import llama_cpp
        log.info("Auto-detected provider: gguf (llama-cpp-python installed)")
        return "gguf"
    except ImportError:
        pass

    log.info("Using default provider: st (sentence-transformers)")
    return "st"


def load_provider(
    provider: str | None = None,
    model: str | None = None,
    model_path: str | None = None,
    batch_size: int = 32,
    **kwargs: Any,
) -> EmbeddingProvider:
    """
    Load an embedding provider by name.

    Args:
        provider: Provider name (st, gguf, openai, cohere, or custom).
                  If None, auto-detects based on environment.
        model: Model name/path (provider-specific).
        model_path: Local model file path (gguf only).
        batch_size: Batch size for document embedding.
        **kwargs: Additional provider-specific arguments.

    Returns:
        Instantiated EmbeddingProvider.
    """
    if provider is None or provider == "auto":
        provider = auto_detect_provider()

    # Try built-in providers first
    if provider in BUILTIN_PROVIDERS:
        module_path, class_name = BUILTIN_PROVIDERS[provider]
        cls = _load_class(module_path, class_name)
    else:
        # Try entry points for custom providers
        cls = _load_entry_point(provider)
        if cls is None:
            raise EmbeddingError(
                f"Unknown provider: {provider}. "
                f"Available: {list(BUILTIN_PROVIDERS.keys())}"
            )

    # Build constructor kwargs based on provider
    init_kwargs: dict[str, Any] = {"batch_size": batch_size}

    if provider == "gguf":
        if model_path:
            init_kwargs["model_path"] = model_path
        elif model:
            from pathlib import Path
            init_kwargs["model_path"] = Path(model)
    elif provider in ("st", "openai", "cohere"):
        if model:
            init_kwargs["model_name"] = model

    init_kwargs.update(kwargs)

    log.info("Loading provider: %s (model=%s)", provider, model or "default")
    return cls(**init_kwargs)
