"""MCP server - primary external interface for the RAG system."""

import asyncio
import json
from typing import Any

from mcp.server.lowlevel.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp import types

from rag.config import get_settings
from rag.logging import get_logger, setup_logging
from rag.mcp.tools import RAGTools
from rag.pipeline import RAGPipeline

log = get_logger("mcp.server")

TOOLS = [
    types.Tool(
        name="rag.search",
        description="Search for relevant document chunks. Agent uses its own model to generate answers from these chunks.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 10, "description": "Number of results"},
                "document_id": {"type": "string", "description": "Filter by document ID"},
                "file_type": {"type": "string", "description": "Filter by file type"},
                "source": {"type": "string", "description": "Filter by source file"},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="rag.ingest",
        description="Convert, chunk, embed, and store a document. Path resolved against allowed input directory.",
        inputSchema={
            "type": "object",
            "properties": {
                "relative_path": {"type": "string", "description": "Relative path to document"},
            },
            "required": ["relative_path"],
        },
    ),
    types.Tool(
        name="rag.get_chunk",
        description="Retrieve full chunk content and metadata for evidence inspection.",
        inputSchema={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "Chunk ID to retrieve"},
            },
            "required": ["chunk_id"],
        },
    ),
    types.Tool(
        name="rag.status",
        description="System health: loaded models, collection stats, config.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="rag.delete",
        description="Remove a document from the index.",
        inputSchema={
            "type": "object",
            "properties": {
                "document_id": {"type": "string", "description": "Document ID to delete"},
            },
            "required": ["document_id"],
        },
    ),
]


def create_server(pipeline: RAGPipeline, rag_tools: RAGTools) -> Server:

    async def handle_list_tools(ctx: ServerRequestContext, request: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=TOOLS)

    async def handle_call_tool(ctx: ServerRequestContext, request: types.CallToolRequestParams) -> types.CallToolResult:
        name = request.name
        arguments = request.arguments or {}
        try:
            if name == "rag.search":
                result = rag_tools.search(**arguments)
            elif name == "rag.ingest":
                result = rag_tools.ingest(**arguments)
            elif name == "rag.get_chunk":
                result = rag_tools.get_chunk(**arguments)
            elif name == "rag.status":
                result = rag_tools.status()
            elif name == "rag.delete":
                result = rag_tools.delete(**arguments)
            else:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True,
                )

            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))],
            )

        except Exception as e:
            log.error("Tool %s failed: %s", name, e)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: {e}")],
                isError=True,
            )

    async def handle_list_resources(ctx: ServerRequestContext, request: Any) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(uri="rag://config", name="RAG Configuration", mimeType="application/json"),
                types.Resource(uri="rag://models", name="Embedding Model Info", mimeType="application/json"),
                types.Resource(uri="rag://stats", name="Index Statistics", mimeType="application/json"),
            ]
        )

    async def handle_read_resource(ctx: ServerRequestContext, request: types.ReadResourceRequestParams) -> types.ReadResourceResult:
        uri = request.uri
        if uri == "rag://config":
            content = json.dumps({
                "chunk_size": pipeline._settings.chunk_size,
                "chunk_overlap": pipeline._settings.chunk_overlap,
                "allowed_input_dir": str(pipeline._settings.allowed_input_dir),
            }, indent=2)
        elif uri == "rag://models":
            content = json.dumps(pipeline._embedding_provider.model_info(), indent=2)
        elif uri == "rag://stats":
            content = json.dumps(pipeline.status(), indent=2)
        else:
            content = f"Unknown resource: {uri}"

        return types.ReadResourceResult(
            contents=[types.ResourceContents(uri=uri, mimeType="application/json", text=content)]
        )

    server = Server(
        "rag-server",
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
        on_list_resources=handle_list_resources,
        on_read_resource=handle_read_resource,
    )

    return server


async def run_server(pipeline: RAGPipeline, tools: RAGTools) -> None:
    server = create_server(pipeline, tools)

    settings = get_settings()
    log.info("Starting MCP server on %s:%d", settings.mcp_host, settings.mcp_port)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def start_server(pipeline: RAGPipeline, tools: RAGTools) -> None:
    asyncio.run(run_server(pipeline, tools))
