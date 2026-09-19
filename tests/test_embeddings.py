"""Tests for embedding providers."""

import pytest

from rag.embeddings.base import EmbeddingProvider
from rag.errors import EmbeddingError


class TestEmbeddingProviderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()


class TestQwenLocalEmbedding:
    def test_missing_model(self, tmp_path):
        from rag.embeddings.qwen_local import QwenLocalEmbedding

        provider = QwenLocalEmbedding(model_path=tmp_path / "nonexistent.gguf")
        with pytest.raises(EmbeddingError, match="Embedding model not found"):
            provider.embed_query("test")

    def test_model_info(self, tmp_path):
        from rag.embeddings.qwen_local import QwenLocalEmbedding

        fake_model = tmp_path / "model.gguf"
        provider = QwenLocalEmbedding(model_path=fake_model, dimension=512, batch_size=16)
        info = provider.model_info()
        assert info["dimension"] == 512
        assert info["batch_size"] == 16

    def test_dimension(self, tmp_path):
        from rag.embeddings.qwen_local import QwenLocalEmbedding

        provider = QwenLocalEmbedding(model_path=tmp_path / "model.gguf", dimension=768)
        assert provider.dimension() == 768
