#!/usr/bin/env bash
# RAG MCP Server - Quick Setup for Linux/macOS
# Usage: ./setup.sh

set -e

echo "========================================"
echo " RAG MCP Server - Installation"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Install Python 3.11+"
    exit 1
fi

echo "[1/3] Installing Python dependencies..."
python3 -m pip install -e . --quiet

echo "[2/3] Creating .env from template..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env - please edit with your settings"
else
    echo ".env already exists, skipping"
fi

echo "[3/3] Testing import..."
python3 -c "from rag.cli import main; print('OK')"

echo ""
echo "========================================"
echo " Installation complete!"
echo "========================================"
echo ""
echo "Usage with opencode:"
echo '  Add to opencode.jsonc:'
echo '  "mcp": {'
echo '    "rag": {'
echo '      "type": "local",'
echo '      "command": ["python3", "-m", "rag.cli", "serve"],'
echo "      \"cwd\": \"$(pwd)\","
echo '      "environment": {'
echo "        \"PYTHONPATH\": \"$(pwd)/src\""
echo '      }'
echo '    }'
echo '  }'
echo ""
echo "Manual usage:"
echo "  python3 -m rag.cli serve        # Start MCP server"
echo "  python3 -m rag.cli status       # Check status"
echo "  python3 -m rag.cli ingest file  # Ingest document"
echo '  python3 -m rag.cli search "query"  # Search'
echo ""
