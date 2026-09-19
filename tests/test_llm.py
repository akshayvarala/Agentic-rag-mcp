"""Tests for LLM provider."""

import pytest

from rag.llm.base import LLMProvider
from rag.errors import LLMError


class TestLLMProviderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LLMProvider()


class TestLlamaCppProvider:
    def test_missing_model(self, tmp_path):
        from rag.llm.llama_cpp import LlamaCppProvider

        provider = LlamaCppProvider(model_path=tmp_path / "nonexistent.gguf")
        with pytest.raises(LLMError, match="LLM model not found"):
            provider.generate("test")

    def test_model_info(self, tmp_path):
        from rag.llm.llama_cpp import LlamaCppProvider

        fake_model = tmp_path / "model.gguf"
        provider = LlamaCppProvider(
            model_path=fake_model,
            n_ctx=2048,
            n_gpu_layers=0,
            temperature=0.5,
        )
        info = provider.model_info()
        assert info["n_ctx"] == 2048
        assert info["n_gpu_layers"] == 0
        assert info["temperature"] == 0.5

    def test_stream_not_implemented(self, tmp_path):
        from rag.llm.llama_cpp import LlamaCppProvider

        provider = LlamaCppProvider(model_path=tmp_path / "model.gguf")
        with pytest.raises(LLMError, match="LLM model not found"):
            list(provider.stream_generate("test"))
