"""llama.cpp LLM provider via llama-cpp-python."""

from pathlib import Path
from typing import Any, Iterator

from rag.errors import LLMError
from rag.llm.base import LLMProvider
from rag.logging import get_logger

log = get_logger("llm.llama_cpp")


class LlamaCppProvider(LLMProvider):
    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.7,
        mmproj_path: Path | None = None,
    ):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._temperature = temperature
        self._mmproj_path = mmproj_path
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        if not self._model_path.exists():
            raise LLMError(f"LLM model not found: {self._model_path}")

        try:
            from llama_cpp import Llama

            log.info("Loading LLM model: %s", self._model_path)
            kwargs: dict[str, Any] = {
                "model_path": str(self._model_path),
                "n_ctx": self._n_ctx,
                "n_gpu_layers": self._n_gpu_layers,
                "verbose": False,
            }
            if self._mmproj_path and self._mmproj_path.exists():
                kwargs["clip_model_path"] = str(self._mmproj_path)

            self._model = Llama(**kwargs)
            log.info("LLM model loaded successfully")
            return self._model
        except ImportError:
            raise LLMError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        except Exception as e:
            raise LLMError(f"Failed to load LLM model: {e}", cause=e)

    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        model = self._load_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            result = model.create_chat_completion(
                messages=messages,
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", 2048),
            )
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"Generation failed: {e}", cause=e)

    def stream_generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> Iterator[str]:
        model = self._load_model()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = model.create_chat_completion(
                messages=messages,
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", 2048),
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    yield delta["content"]
        except Exception as e:
            raise LLMError(f"Streaming generation failed: {e}", cause=e)

    def model_info(self) -> dict[str, Any]:
        return {
            "model_path": str(self._model_path),
            "mmproj_path": str(self._mmproj_path) if self._mmproj_path else None,
            "n_ctx": self._n_ctx,
            "n_gpu_layers": self._n_gpu_layers,
            "temperature": self._temperature,
        }
