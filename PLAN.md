# RAG System — Implementation Plan

## Goal

Build a local-first RAG **retrieval engine** designed to be consumed as a **plugin for agentic environments** (Hermes, opencode, etc.) via MCP.

**The agent's own model handles generation.** The RAG system retrieves relevant context — the agent reasons and answers. No redundant LLM loaded by the plugin.

## Current State Assessment

| Component | Status |
|-----------|--------|
| Python 3.14.3 | Installed |
| torch 2.11.0 | Installed |
| transformers 5.4.0 | Installed |
| llama.cpp binaries | Installed at `C:\llama-cpp\` |
| Qwen3-Embedding-0.6B-Q8_0.gguf | Available — 639MB embedding model |
| Qwen3.5-0.8B-UD-Q4_K_XL.gguf | Available — 559MB LLM model |
| mmproj-BF16.gguf | Available — 207MB multimodal projection (vision, later milestone) |
| llama-cpp-python | **MISSING** — needed to load GGUF models from Python |
| qdrant-client | **MISSING** — needed for local vector database |
| markitdown | **MISSING** — needed for document conversion |
| sentence-transformers | **MISSING** — optional, for fallback embedding loading |
| C:\RAG | Empty — clean slate |

## Architecture

```
C:\RAG\
├── pyproject.toml                 # Project metadata & dependencies
├── .env.example                   # Configuration template
├── .gitignore
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── config.py              # Pydantic settings from env
│       ├── logging.py             # Structured logging configuration
│       ├── converters/
│       │   ├── __init__.py
│       │   ├── base.py            # DocumentConverter ABC
│       │   └── markitdown.py      # MarkItDown implementation
│       ├── chunking/
│       │   ├── __init__.py
│       │   └── chunker.py         # Text chunking with metadata
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py            # EmbeddingProvider ABC
│       │   └── qwen_local.py      # Qwen GGUF via llama-cpp-python
│       ├── llm/                      # OPTIONAL — standalone CLI only, not used by agent
│       │   ├── __init__.py
│       │   ├── base.py               # LLMProvider ABC
│       │   └── llama_cpp.py          # llama.cpp via llama-cpp-python
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   ├── base.py            # VectorStore ABC
│       │   └── qdrant_local.py    # Qdrant local implementation
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── retriever.py       # Retrieval logic
│       │   └── reranker.py        # Reranking ABC + implementations
│       ├── pipeline.py            # Full RAG pipeline orchestration
│       ├── metadata.py            # Indexing metadata tracking
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── server.py          # MCP server (primary external interface)
│       │   ├── security.py        # Path sandboxing, input validation
│       │   └── tools.py           # MCP tool definitions (ingest, query, etc.)
│       ├── agent/
│       │   ├── __init__.py
│       │   └── adapter.py         # Thin adapter for agentic env integration
│       └── cli.py                 # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_converters.py
│   ├── test_embeddings.py
│   ├── test_vectorstore.py
│   ├── test_retrieval.py
│   ├── test_reranker.py
│   ├── test_mcp.py
│   ├── test_pipeline.py
│   ├── test_security.py
│   └── test_llm.py            # Optional — standalone LLM tests
└── data/
    ├── original/
    ├── normalized/
    ├── metadata/
    ├── qdrant/                    # Qdrant local persistence
    └── assets/
        ├── figures/
        ├── images/
        └── tables/
```

## Implementation Steps

### Step 1: Project Setup

Create `pyproject.toml`, `.env.example`, `.gitignore`.

**Dependencies to install:**
```
llama-cpp-python>=0.3.0    # Embeddings only (required). LLM optional, not loaded in plugin mode.
qdrant-client>=1.12.0      # Local vector database
markitdown[microsoft]      # Document-to-markdown conversion
python-dotenv              # Env config loading
pydantic>=2.0              # Settings validation
pydantic-settings          # BaseSettings for env loading
mcp>=1.0                   # MCP server framework
```

**Configuration (`.env.example`):**
```env
# Embedding (required)
EMBEDDING_MODEL_PATH=C:\llama-cpp\Qwen3-Embedding-0.6B-Q8_0.gguf
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=32

# LLM (optional — only for standalone CLI mode, not used by agent/plugin)
LLAMA_CPP_MODEL_PATH=C:\llama-cpp\Qwen3.5-0.8B-UD-Q4_K_XL.gguf
LLAMA_CPP_MMPROJ_PATH=C:\llama-cpp\mmproj-BF16.gguf
LLAMA_CPP_N_CTX=4096
LLAMA_CPP_N_GPU_LAYERS=-1
LLAMA_CPP_TEMPERATURE=0.7

# Vector Store (required)
QDRANT_PATH=./data/qdrant
QDRANT_COLLECTION=rag_documents

# Pipeline (required)
DATA_DIR=./data
ALLOWED_INPUT_DIR=./data/original
CHUNK_SIZE=1024
CHUNK_OVERLAP=128

# MCP Server (required)
MCP_HOST=127.0.0.1
MCP_PORT=3000
LOG_LEVEL=INFO
```

### Step 2: Configuration & Logging (`config.py`, `logging.py`)

- Pydantic `BaseSettings` with `env_file=".env"` loading
- All paths as `Path` type with validation
- `EMBEDDING_MODEL_PATH` points to local Qwen GGUF (required)
- `LLAMA_CPP_MODEL_PATH` points to local LLM GGUF (optional, standalone only)
- Record embedding model name/path for reproducibility
- Structured logging with `LOG_LEVEL` config
- Log format includes: timestamp, component, model path, operation, duration
- Separate loggers for: pipeline, embeddings, MCP

### Step 3: Document Conversion (`converters/`)

- `DocumentConverter` ABC with `convert(input_path) -> ConvertedDocument`
- `MarkItDownConverter` implementation using MarkItDown Python API
- Output structure: original + normalized markdown + metadata JSON + assets
- Input constrained to `ALLOWED_INPUT_DIR` for security
- Metadata preserved: document_id, source_file, file_type, headings, etc.

### Step 4: Chunking (`chunking/`)

- Text chunking with configurable chunk_size and overlap
- Preserves heading hierarchy metadata
- Each chunk gets unique ID, position info, source document reference

### Step 5: Embeddings (`embeddings/`)

- `EmbeddingProvider` ABC with `embed_documents()` and `embed_query()`
- `QwenLocalEmbedding` implementation:
  - Loads Qwen GGUF via `llama_cpp.Llama(model_path=..., embedding=True)`
  - **Separate methods** for document vs query embedding (Qwen3 is instruction-aware):
    - `embed_documents(texts)` → prepends task-specific instruction prefix for documents
    - `embed_query(query)` → prepends task-specific instruction prefix for queries
  - Different prefixes produce measurably different embeddings — do NOT treat both inputs identically
  - Configurable batch size for document embedding
  - Records model path in index metadata for reproducibility

### Step 6: Optional LLM Provider (`llm/`) — Standalone CLI Only

**Not used in agent/plugin mode.** The agent's own model handles generation. This module exists only for standalone CLI usage.

- `LLMProvider` ABC with:
  - `generate(prompt, system_prompt, **kwargs) -> str`
  - `stream_generate(prompt, system_prompt, **kwargs) -> Iterator[str]` (optional)
  - `model_info() -> dict` — returns model name, path, context size, parameter count
- `LlamaCppProvider` implementation:
  - Loads `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` via `llama_cpp.Llama(model_path=...)`
  - Optional multimodal support via `mmproj-BF16.gguf` (later milestone)
  - Configurable context size, GPU layers, temperature
  - System prompt injection for RAG context
  - `model_info()` returns loaded model metadata

### Step 7: Vector Store (`vectorstore/`)

- `VectorStore` ABC with `upsert()`, `search()`, `delete()`, `create_collection()`
- `QdrantLocalStore` implementation:
  - Uses Qdrant in-memory or local file storage
  - Records embedding model path in collection metadata
  - Cosine similarity search with configurable top-k

### Step 8: Retrieval & Reranking (`retrieval/`)

- `Retriever` class that:
  - Embeds query using EmbeddingProvider.embed_query()
  - Searches VectorStore
  - Applies optional reranker
  - Returns ranked chunks with scores
- `Reranker` ABC with `rerank(query, chunks) -> ranked_chunks`

**Reranker milestones:**
- **MVP:** `NoopReranker` — pass-through, no reranking
- **Milestone 2:** `LocalReranker` — Qwen3-Reranker-0.6B or another local cross-encoder
- **Experiment:** Dense retrieval vs Dense + Reranker — measure impact on retrieval quality

### Step 9: Pipeline (`pipeline.py`)

- Full orchestration: convert → chunk → embed → store → search
- No generation step — the agent does that with its own model
- Each step uses the abstract interfaces
- Pipeline metadata tracks: embedding model, chunk config, timestamps
- Structured error handling at each stage with clear error messages
- Pipeline class exposes both sync and async interfaces

### Step 10: Error Handling

- Custom exception hierarchy:
  - `RAGError` (base)
  - `ConversionError` — document conversion failures
  - `EmbeddingError` — embedding generation failures
  - `VectorStoreError` — Qdrant operation failures
  - `SecurityError` — path sandbox violations
  - `ConfigError` — configuration/missing model errors
  - `LLMError` — inference failures (optional, standalone mode only)
- Each exception carries: error code, component name, human message, original cause
- Pipeline catches and wraps lower-level errors with context

### Step 11: MCP Server (`mcp/`)

The MCP server is the **primary external interface** — this is how agents interact with the RAG system.

**MCP Tools exposed:**
- `rag.search(query, top_k, document_id=None, file_type=None, source=None)` — **Primary tool.** Retrieve ranked chunks with scores + metadata. Supports metadata filtering. Agent uses its own model to generate answers from these chunks.
- `rag.ingest(relative_path)` — Convert, chunk, embed, store a document. Path resolved against `ALLOWED_INPUT_DIR`. Rejects absolute paths and `..` escapes.
- `rag.get_chunk(chunk_id)` — Retrieve full chunk content + metadata for evidence inspection. Separates retrieval from inspection.
- `rag.status()` — System health: loaded models, collection stats, config
- `rag.delete(document_id)` — Remove a document from the index

**MCP Resources exposed:**
- `rag://config` — Current configuration (read-only)
- `rag://models` — Embedding model info (read-only)
- `rag://stats` — Index statistics

**MCP Security (`mcp/security.py`):**
- Path sandboxing: all file operations resolved against `ALLOWED_INPUT_DIR`
- `rag.ingest()` accepts relative paths only — rejects absolute paths
- Input validation: reject paths with `..`, symlinks outside workspace, null bytes
- Resource limits: max file size, max files per ingest, max query length
- Audit logging: log all tool invocations with timestamps
- No arbitrary filesystem access — only pre-approved directories

### Step 12: Agent Adapter (`agent/adapter.py`)

Thin adapter for integration with agentic environments (Hermes, opencode):
- Wraps MCP server as a library call (no subprocess needed)
- Provides `RAGAgent` class with methods matching MCP tools
- Manages lifecycle: init, health check, shutdown
- Plugin registration interface for agent frameworks

### Step 13: CLI Entry Point (`cli.py`)

- `rag ingest <relative_path>` — Ingest document (resolved against ALLOWED_INPUT_DIR)
- `rag search "query"` — Search with optional metadata filters
- `rag get-chunk <chunk_id>` — Inspect a specific chunk
- `rag status` — Show system status
- `rag serve` — Start MCP server
- Uses same config as MCP server (reads `.env`)
- Note: no `rag query` — use the agent's model for generation

### Step 14: Tests

- Unit tests for each component
- Security tests: path traversal, symlink escape, oversized files, absolute path rejection
- MCP tool tests: verify each tool's contract (ingest, search, get_chunk, status, delete)
- Metadata filtering tests: search with document_id, file_type, source filters
- Integration test: ingest document → search → get_chunk → verify evidence
- Test that all ABCs are properly abstract (can't instantiate directly)
- Optional LLM tests for standalone CLI mode

## Key Design Decisions

1. **Retrieval-only architecture** — The RAG plugin retrieves context. The agent's own model generates answers. No redundant LLM loaded. Saves ~559MB memory and avoids model confusion.

2. **MCP-first interface** — The MCP server is the primary external interface, not a REST API. This makes the RAG system directly consumable by agentic environments (Hermes, opencode) without extra adapters.

3. **Instruction-aware embeddings** — Qwen3-Embedding is instruction-aware. `embed_documents()` and `embed_query()` use different task-specific prefixes. Measurably different embeddings — do not treat both inputs identically.

4. **Separate retrieval and inspection** — `rag.search()` finds chunks, `rag.get_chunk()` inspects evidence. Agent explicitly chooses what to read deeply.

5. **Relative path sandboxing** — `rag.ingest()` accepts relative paths resolved against `ALLOWED_INPUT_DIR`. No absolute paths, no `..` escapes. MarkItDown's I/O privileges are sandboxed.

6. **Metadata filtering** — `rag.search()` supports filters by document_id, file_type, source. Agent can narrow search to specific document types or sources.

7. **Reranker milestones** — MVP uses NoopReranker. Milestone 2 adds local cross-encoder (Qwen3-Reranker-0.6B). Experiment: Dense vs Dense + Reranker.

8. **Qdrant local mode** — No Docker required, runs in-process or with local file persistence.

9. **MarkItDown as sole converter** — No custom PDF/DOCX parsers; MarkItDown handles all formats.

10. **Plugin-ready design** — Every component is behind an ABC so the system can be extended or swapped without touching core pipeline code.

## Missing Items to Install

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| llama-cpp-python | GGUF embedding model loading (required) | `pip install llama-cpp-python` |
| qdrant-client | Local vector database | `pip install qdrant-client` |
| markitdown | Document conversion | `pip install markitdown[microsoft]` |
| python-dotenv | Config loading | `pip install python-dotenv` |
| pydantic-settings | Settings validation | `pip install pydantic-settings` |
| mcp | MCP server framework | `pip install mcp` |

## Potential Risks

1. **Qwen3 GGUF embedding prompt format** — Need to verify exact instruction format for Qwen3-Embedding-0.6B. Different embedding models have different prompt templates.

2. **llama-cpp-python CUDA build** — Since C:\llama-cpp has CUDA DLLs, llama-cpp-python needs CUDA support. May need `CMAKE_ARGS="-DGGML_CUDA=on"` during install, or use prebuilt wheel.

3. **Qwen embedding dimension** — Need to confirm the actual dimension (likely 1024 for 0.6B model) and set Qdrant collection config accordingly.

4. **MCP SDK compatibility** — The `mcp` Python SDK may have breaking changes. Pin version and test tool registration thoroughly.

5. **Agent framework integration** — Hermes/opencode plugin formats may differ. Keep the adapter thin so it's easy to adjust.

## Local Model Inventory

| File | Size | Purpose | Required |
|------|------|---------|----------|
| `Qwen3-Embedding-0.6B-Q8_0.gguf` | 639MB | Embedding model (Q8_0 quantization) | **Yes** |
| `Qwen3.5-0.8B-UD-Q4_K_XL.gguf` | 559MB | LLM for generation (standalone CLI only) | Optional |
| `mmproj-BF16.gguf` | 207MB | Multimodal projection — vision, later milestone | Optional |

All models at `C:\llama-cpp\`. No downloads needed. No cloud APIs.
Only the embedding model is loaded in agent/plugin mode (~639MB vs ~1.2GB).
