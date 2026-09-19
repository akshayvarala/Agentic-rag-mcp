# Agentic-RAG-MCP

Local-first RAG **retrieval engine** built as a plugin for agentic environments via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io).

> **The agent's own model handles generation.** This server retrieves relevant context — the agent reasons and answers. No redundant LLM loaded by the plugin.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-2.x-green.svg)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-51%20passing-brightgreen.svg)](#testing)

## Why

Agentic coding tools (opencode, Claude Desktop, Hermes, pi, …) are great at reasoning but have no memory of *your* documents. Agentic-RAG-MCP fills that gap:

- **Ingest** PDFs, DOCX, code, notes → normalized Markdown → chunks → embeddings → local vector DB
- **Search** via MCP tools with metadata filtering — the agent grounds answers in your data
- **Inspect** exact evidence with `rag.get_chunk` before answering

Everything runs locally. No cloud APIs, no data leaves your machine.

## Features

| Area | Details |
|------|---------|
| **MCP-native** | 5 tools + 3 resources, works with opencode, Claude Desktop, any MCP client |
| **Instruction-aware embeddings** | Qwen3-Embedding with separate document/query prefixes |
| **Pluggable backends** | Embedding registry: local GGUF, SentenceTransformers, OpenAI, Cohere |
| **Local vector DB** | Qdrant in-process/file persistence, cosine similarity |
| **Sandboxed ingestion** | Relative paths only; rejects absolute paths, `..`, null bytes |
| **Metadata filtering** | Filter search by `document_id`, `file_type`, `source` |
| **ABC everywhere** | Converter, chunker, embeddings, vector store, reranker, LLM all swappable |

## Quick Start

```bash
git clone https://github.com/akshayvarala/Agentic-rag-mcp.git
cd Agentic-rag-mcp
pip install -e ".[gguf]"
cp .env.example .env   # edit model paths
```

```bash
python -m rag.cli status
python -m rag.cli ingest notes.pdf
python -m rag.cli search "what is retrieval augmented generation?"
python -m rag.cli serve   # start MCP server (stdio)
```

## MCP Integration

### opencode

`~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "rag": {
      "type": "local",
      "command": ["python", "-m", "rag.cli", "serve"],
      "cwd": "/path/to/Agentic-rag-mcp",
      "environment": { "PYTHONPATH": "/path/to/Agentic-rag-mcp/src" },
      "enabled": true
    }
  }
}
```

### Claude Desktop

`claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag": {
      "command": "python",
      "args": ["-m", "rag.cli", "serve"],
      "cwd": "/path/to/Agentic-rag-mcp",
      "env": { "PYTHONPATH": "/path/to/Agentic-rag-mcp/src" }
    }
  }
}
```

### Any MCP client (stdio)

```bash
PYTHONPATH=src python -m rag.cli serve
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `rag.search(query, top_k, document_id?, file_type?, source?)` | **Primary tool.** Ranked chunks with scores + metadata |
| `rag.ingest(relative_path)` | Convert → chunk → embed → store. Sandboxed to `ALLOWED_INPUT_DIR` |
| `rag.get_chunk(chunk_id)` | Full chunk text + metadata for evidence inspection |
| `rag.status()` | Models, collection stats, config |
| `rag.delete(document_id)` | Remove a document from the index |

Resources: `rag://config`, `rag://models`, `rag://stats`.

## Configuration

`.env` (see `.env.example`):

```env
# Embeddings (required) — pick ONE provider
EMBEDDING_PROVIDER=qwen_local
EMBEDDING_MODEL_PATH=C:\llama-cpp\Qwen3-Embedding-0.6B-Q8_0.gguf
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=32

# Vector store
QDRANT_PATH=./data/qdrant
QDRANT_COLLECTION=rag_documents

# Pipeline
DATA_DIR=./data
ALLOWED_INPUT_DIR=./data/original
CHUNK_SIZE=1024
CHUNK_OVERLAP=128

# MCP
MCP_HOST=127.0.0.1
MCP_PORT=3000
LOG_LEVEL=INFO
```

Embedding providers (`EMBEDDING_PROVIDER`): `qwen_local` (GGUF via llama-cpp),
`st_local` (SentenceTransformers), `openai`, `cohere`.

## Architecture

```
src/rag/
├── config.py          # Pydantic BaseSettings from .env
├── logging.py         # Structured logging (stderr-safe for MCP stdio)
├── errors.py          # RAGError + Conversion/Embedding/VectorStore/Security/Config/LLM/...
├── metadata.py        # Index metadata tracking
├── pipeline.py        # convert → chunk → embed → store → search
├── cli.py             # ingest / search / get-chunk / status / serve
├── converters/        # DocumentConverter ABC + MarkItDown
├── chunking/          # Overlapping text chunker with positions
├── embeddings/        # base + qwen_local + st_local + openai + cohere + registry
├── llm/               # Optional standalone LLM (llama-cpp) — NOT used by agent
├── vectorstore/       # VectorStore ABC + Qdrant local
├── retrieval/         # Retriever + Reranker ABC
├── mcp/               # server (MCP 2.x API) + tools + security
└── agent/             # Thin RAGAgent adapter for harnesses
```

Key design decisions:

1. **Retrieval-only** — saves ~559MB vs loading a second LLM; avoids model confusion.
2. **Instruction-aware embeddings** — `embed_documents()` and `embed_query()` use different prefixes; never treat them identically.
3. **Separate search/inspect** — `rag.search` finds, `rag.get_chunk` reads deeply.
4. **MCP-first** — no REST layer; agents consume tools directly.

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -q
# 51 passed
```

Covers: chunking, converters, embeddings, vector store, retrieval, reranker,
security (traversal/absolute-path/null-byte/oversize), pipeline, MCP tools, LLM.

## Requirements

- Python 3.11+
- A GGUF embedding model **or** one of the API providers
- ~1GB disk for Qdrant data (grows with corpus)

## Roadmap

- [ ] `LocalReranker` (Qwen3-Reranker-0.6B cross-encoder)
- [ ] Dense vs Dense+Reranker quality experiment
- [ ] Docker image + `uvx` one-liner
- [ ] Vision milestone via `mmproj` multimodal projection

## License

MIT — see [LICENSE](LICENSE).

## Contributors

- [@akshayvarala](https://github.com/akshayvarala) — owner
- [@manojkumar9121](https://github.com/manojkumar9121) — contributor
