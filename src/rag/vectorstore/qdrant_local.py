"""Qdrant local vector store implementation."""

import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag.errors import VectorStoreError
from rag.logging import get_logger
from rag.vectorstore.base import SearchResult, VectorStore

log = get_logger("vectorstore.qdrant")


class QdrantLocalStore(VectorStore):
    def __init__(self, path: Path, collection: str = "rag_documents"):
        self._path = path
        self._collection = collection
        self._client: QdrantClient | None = None

    def _get_client(self) -> QdrantClient:
        if self._client is not None:
            return self._client

        try:
            self._path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self._path))
            log.info("Connected to Qdrant at %s", self._path)
            return self._client
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to Qdrant: {e}", cause=e)

    def create_collection(self, dimension: int) -> None:
        client = self._get_client()
        try:
            collections = client.get_collections().collections
            names = [c.name for c in collections]

            if self._collection in names:
                client.delete_collection(self._collection)

            client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            log.info("Created collection '%s' with dimension %d", self._collection, dimension)
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Failed to create collection: {e}", cause=e)

    def _to_uuid(self, id_str: str) -> str:
        try:
            uuid.UUID(id_str)
            return id_str
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))

    def upsert(self, ids: list[str], vectors: list[list[float]], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        client = self._get_client()
        points = []
        for i, (vec_id, vector, text, meta) in enumerate(zip(ids, vectors, texts, metadatas)):
            payload = {"text": text, **meta}
            points.append(PointStruct(id=self._to_uuid(vec_id), vector=vector, payload=payload))

        try:
            client.upsert(collection_name=self._collection, points=points)
            log.info("Upserted %d points", len(points))
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert: {e}", cause=e)

    def search(self, query_vector: list[float], top_k: int = 10, filters: dict[str, Any] | None = None) -> list[SearchResult]:
        client = self._get_client()

        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            query_filter = Filter(must=conditions)

        try:
            results = client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
            )

            return [
                SearchResult(
                    id=str(hit.id),
                    text=hit.payload.get("text", ""),
                    score=hit.score,
                    metadata={k: v for k, v in hit.payload.items() if k != "text"},
                )
                for hit in results.points
            ]
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}", cause=e)

    def delete(self, ids: list[str]) -> None:
        client = self._get_client()
        try:
            uuid_ids = [self._to_uuid(id_str) for id_str in ids]
            client.delete(collection_name=self._collection, points_selector=uuid_ids)
            log.info("Deleted %d points", len(ids))
        except Exception as e:
            raise VectorStoreError(f"Delete failed: {e}", cause=e)

    def delete_by_filter(self, filters: dict[str, Any]) -> None:
        client = self._get_client()
        conditions = [FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filters.items()]
        try:
            client.delete(
                collection_name=self._collection,
                points_selector=Filter(must=conditions),
            )
            log.info("Deleted by filter: %s", filters)
        except Exception as e:
            raise VectorStoreError(f"Delete by filter failed: {e}", cause=e)

    def count(self) -> int:
        client = self._get_client()
        try:
            info = client.get_collection(self._collection)
            return info.points_count or 0
        except Exception as e:
            raise VectorStoreError(f"Count failed: {e}", cause=e)

    def collection_info(self) -> dict[str, Any]:
        client = self._get_client()
        try:
            info = client.get_collection(self._collection)
            result: dict[str, Any] = {
                "name": self._collection,
                "points_count": info.points_count,
                "config": str(info.config),
            }
            return result
        except Exception as e:
            raise VectorStoreError(f"Collection info failed: {e}", cause=e)
