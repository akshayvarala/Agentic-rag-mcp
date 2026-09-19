@echo off
REM RAG MCP Server - Quick Setup for Windows
REM Usage: install.bat

echo ========================================
echo  RAG MCP Server - Installation
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ from https://python.org
    exit /b 1
)

echo [1/3] Installing Python dependencies...
python -m pip install -e . --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo [2/3] Creating .env from template...
if not exist .env (
    copy .env.example .env >nul
    echo Created .env - please edit with your settings
) else (
    echo .env already exists, skipping
)

echo [3/3] Testing import...
python -c "from rag.cli import main; print('OK')" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Import failed
    exit /b 1
)

echo.
echo ========================================
echo  Installation complete!
echo ========================================
echo.
echo Usage with opencode:
echo   Add to opencode.jsonc:
echo   "mcp": {
echo     "rag": {
echo       "type": "local",
echo       "command": ["python", "-m", "rag.cli", "serve"],
echo       "cwd": "%CD%",
echo       "environment": {
echo         "PYTHONPATH": "%CD%\\src"
echo       }
echo     }
echo   }
echo.
echo Manual usage:
echo   python -m rag.cli serve        # Start MCP server
echo   python -m rag.cli status       # Check status
echo   python -m rag.cli ingest file  # Ingest document
echo   python -m rag.cli search "query"  # Search
echo.
