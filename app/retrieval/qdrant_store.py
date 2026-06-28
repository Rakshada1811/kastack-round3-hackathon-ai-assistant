from __future__ import annotations

import os
from typing import Any

import numpy as np

from app.ingestion import MemoryDocument


class QdrantVectorStore:
    """Optional Qdrant adapter used when qdrant-client and credentials are available."""

    def __init__(self, collection_name: str = "conversation_memory") -> None:
        self.collection_name = os.getenv("QDRANT_COLLECTION", collection_name)
        self.client: Any | None = None
        self.enabled = False
        self._connect()

    def _connect(self) -> None:
        try:
            from qdrant_client import QdrantClient

            url = os.getenv("QDRANT_URL")
            api_key = os.getenv("QDRANT_API_KEY")
            if url:
                self.client = QdrantClient(url=url, api_key=api_key)
            elif os.getenv("USE_QDRANT_MEMORY", "0").lower() in {"1", "true", "yes"}:
                self.client = QdrantClient(":memory:")
            else:
                return
            self.enabled = True
        except Exception:
            self.client = None
            self.enabled = False

    def recreate_collection(self, vector_size: int) -> None:
        if not self.enabled or not self.client:
            return

        from qdrant_client.models import Distance, VectorParams

        existing = [collection.name for collection in self.client.get_collections().collections]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(self, documents: list[MemoryDocument], vectors: np.ndarray) -> None:
        if not self.enabled or not self.client:
            return

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=idx,
                vector=vectors[idx].tolist(),
                payload={
                    "document_id": document.id,
                    "text": document.text,
                    "memory_type": document.memory_type,
                    "metadata": document.metadata,
                },
            )
            for idx, document in enumerate(documents)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.enabled or not self.client:
            return []

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector.ravel().tolist(),
            limit=top_k,
            with_payload=True,
        )
        return [{"score": hit.score, "payload": hit.payload or {}} for hit in hits]
