"""CLI entry point for the RAG system."""

import argparse
import json
import sys
from pathlib import Path

from rag.config import get_settings
from rag.converters.markitdown import MarkItDownConverter
from rag.chunking.chunker import TextChunker
from rag.embeddings.registry import load_provider
from rag.logging import get_logger, setup_logging
from rag.pipeline import RAGPipeline
from rag.vectorstore.qdrant_local import QdrantLocalStore

log = get_logger("cli")


def build_pipeline() -> RAGPipeline:
    settings = get_settings()

    converter = MarkItDownConverter()
    chunker = TextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    embedding_provider = load_provider(
        provider=settings.embedding_provider,
        model=settings.embedding_model,
        model_path=settings.embedding_model_path,
        batch_size=settings.embedding_batch_size,
    )

    vector_store = QdrantLocalStore(path=settings.qdrant_path, collection=settings.qdrant_collection)

    return RAGPipeline(
        converter=converter,
        chunker=chunker,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        settings=settings,
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = build_pipeline()
    input_path = Path(args.path)
    result = pipeline.ingest(input_path)
    print(json.dumps(result, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    pipeline = build_pipeline()
    filters = {}
    if args.document_id:
        filters["document_id"] = args.document_id
    if args.file_type:
        filters["file_type"] = args.file_type

    result = pipeline.search(args.query, top_k=args.top_k, filters=filters or None)
    print(json.dumps(result, indent=2, default=str))


def cmd_get_chunk(args: argparse.Namespace) -> None:
    pipeline = build_pipeline()
    result = pipeline.get_chunk(args.chunk_id)
    if result:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Chunk not found: {args.chunk_id}", file=sys.stderr)
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    pipeline = build_pipeline()
    result = pipeline.status()
    print(json.dumps(result, indent=2, default=str))


def cmd_models(args: argparse.Namespace) -> None:
    from rag.model_downloader import list_available_models

    models = list_available_models()
    print(json.dumps(models, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    from rag.mcp.security import validate_path
    from rag.mcp.server import start_server
    from rag.mcp.tools import RAGTools

    settings = get_settings()
    setup_logging(settings.log_level)

    pipeline = build_pipeline()
    tools = RAGTools(pipeline, settings.allowed_input_dir)

    print(f"Starting MCP server on {settings.mcp_host}:{settings.mcp_port}", file=sys.stderr)
    start_server(pipeline, tools)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Local RAG retrieval engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("path", help="Path to document (absolute or relative to allowed input dir)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = subparsers.add_parser("search", help="Search for relevant chunks")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=10, help="Number of results")
    p_search.add_argument("--document-id", help="Filter by document ID")
    p_search.add_argument("--file-type", help="Filter by file type")
    p_search.set_defaults(func=cmd_search)

    p_get_chunk = subparsers.add_parser("get-chunk", help="Get a specific chunk")
    p_get_chunk.add_argument("chunk_id", help="Chunk ID")
    p_get_chunk.set_defaults(func=cmd_get_chunk)

    p_status = subparsers.add_parser("status", help="Show system status")
    p_status.set_defaults(func=cmd_status)

    p_models = subparsers.add_parser("models", help="List available embedding models")
    p_models.set_defaults(func=cmd_models)

    p_serve = subparsers.add_parser("serve", help="Start MCP server")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    setup_logging()
    args.func(args)


if __name__ == "__main__":
    main()
