"""Custom exception hierarchy for the RAG system."""


class RAGError(Exception):
    """Base exception for all RAG errors."""

    def __init__(self, message: str, component: str = "", code: str = "", cause: Exception | None = None):
        self.message = message
        self.component = component
        self.code = code
        self.__cause__ = cause
        super().__init__(self.message)


class ConversionError(RAGError):
    """Document conversion failures."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, component="converter", code="CONVERSION_ERROR", cause=cause)


class EmbeddingError(RAGError):
    """Embedding generation failures."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, component="embeddings", code="EMBEDDING_ERROR", cause=cause)


class VectorStoreError(RAGError):
    """Qdrant operation failures."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, component="vectorstore", code="VECTORSTORE_ERROR", cause=cause)


class SecurityError(RAGError):
    """Path sandbox violations."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, component="security", code="SECURITY_ERROR", cause=cause)


class ConfigError(RAGError):
    """Configuration or missing model errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, component="config", code="CONFIG_ERROR", cause=cause)


class LLMError(RAGError):
    """Inference failures (standalone mode only)."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, component="llm", code="LLM_ERROR", cause=cause)


class ChunkingError(RAGError):
    """Text chunking failures."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, component="chunking", code="CHUNKING_ERROR", cause=cause)


class RetrievalError(RAGError):
    """Retrieval failures."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, component="retrieval", code="RETRIEVAL_ERROR", cause=cause)
