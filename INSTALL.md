# RAG MCP Server

Local-first RAG retrieval engine for agentic environments via MCP.

## Quick Install

### Windows
```batch
git clone <repo> && cd RAG
install.bat
```

### Linux/macOS
```bash
git clone <repo> && cd RAG
chmod +x setup.sh && ./setup.sh
```

### Manual
```bash
pip install -e .
cp .env.example .env  # edit with your settings
```

## Agent Harness Configurations

### opencode
Add to `~/.config/opencode/opencode.jsonc`:
```jsonc
{
  "mcp": {
    "rag": {
      "type": "local",
      "command": ["python", "-m", "rag.cli", "serve"],
      "cwd": "/path/to/RAG",
      "environment": {
        "PYTHONPATH": "/path/to/RAG/src"
      }
    }
  }
}
```

### Claude Desktop
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "rag": {
      "command": "python",
      "args": ["-m", "rag.cli", "serve"],
      "cwd": "/path/to/RAG",
      "env": {
        "PYTHONPATH": "/path/to/RAG/src"
      }
    }
  }
}
```

### Any MCP Client (generic)
```bash
# Start the MCP server on stdio
python -m rag.cli serve
```

## Manual Usage

```bash
# Check status
python -m rag.cli status

# Ingest a document
python -m rag.cli ingest document.pdf

# Search
python -m rag.cli search "what is RAG?"

# Get specific chunk
python -m rag.cli get-chunk <chunk_id>
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
# Required - paths to local GGUF models
EMBEDDING_MODEL_PATH=C:\llama-cpp\Qwen3-Embedding-0.6B-Q8_0.gguf

# Optional - for standalone CLI mode only
LLAMA_CPP_MODEL_PATH=C:\llama-cpp\Qwen3.5-0.8B-UD-Q4_K_XL.gguf
```

## MCP Tools Exposed

| Tool | Description |
|------|-------------|
| `rag.search` | Search for relevant document chunks |
| `rag.ingest` | Convert, chunk, embed, store a document |
| `rag.get_chunk` | Retrieve full chunk content for inspection |
| `rag.status` | System health and stats |
| `rag.delete` | Remove a document from index |

## Requirements

- Python 3.11+
- GGUF models at `C:\llama-cpp\` (or configure paths in `.env`)
