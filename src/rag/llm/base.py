"""Base LLM provider ABC."""

from abc import ABC, abstractmethod
from typing import Any, Iterator


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> str:
        """Generate a response from a prompt."""
        ...

    def stream_generate(self, prompt: str, system_prompt: str = "", **kwargs: Any) -> Iterator[str]:
        raise NotImplementedError("Streaming not implemented")

    @abstractmethod
    def model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        ...
