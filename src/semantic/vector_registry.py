"""Vector registry that bridges Qdrant collections and PGVector tagging metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    _QDRANT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore
    _QDRANT_AVAILABLE = False


@dataclass
class VectorRecord:
    """Representation of a semantic chunk stored for Fuse.js."""

    id: str
    title: str
    content: str
    url: Optional[str]
    topic: Optional[str]
    tags: List[str] = field(default_factory=list)
    score_hint: Optional[Dict[str, Any]] = None


class VectorRegistry:
    """Manages vector persistence to Qdrant while tracking metadata tags."""

    def __init__(
        self,
        collection: str = "topology_docs",
        qdrant_url: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> None:
        self.collection = collection
        self.qdrant_url = qdrant_url
        self.vector_size = vector_size
        self._records: List[VectorRecord] = []
        self._client: Optional[QdrantClient] = None

        if _QDRANT_AVAILABLE and qdrant_url:
            try:  # pragma: no cover - requires external service
                self._client = QdrantClient(url=qdrant_url)
                logger.info("Connected to Qdrant at %s", qdrant_url)
                if vector_size is not None:
                    self.ensure_collection(vector_size)
            except Exception as exc:  # pragma: no cover
                logger.warning("Unable to connect to Qdrant (%s). Falling back to in-memory store.", exc)
                self._client = None
        else:
            if qdrant_url:
                logger.warning("qdrant-client not installed. Using in-memory registry only.")

    @property
    def records(self) -> List[VectorRecord]:
        return list(self._records)

    def ensure_collection(self, vector_size: int) -> None:
        """Ensure the configured collection exists in Qdrant."""

        if self._client is None or not _QDRANT_AVAILABLE:  # pragma: no cover - optional
            self.vector_size = vector_size
            return

        self.vector_size = vector_size
        try:  # pragma: no cover - requires external service
            collections = self._client.get_collections()
            existing = {collection.name for collection in collections.collections}
            if self.collection not in existing:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Unable to ensure Qdrant collection: %s", exc)
            self._client = None

    def register(
        self,
        chunk_id: str,
        vector: Optional[List[float]],
        payload: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> None:
        """Register a chunk and upsert into Qdrant if possible."""

        record = VectorRecord(
            id=chunk_id,
            title=str(payload.get("title", "")),
            content=str(payload.get("content", "")),
            url=payload.get("url"),
            topic=payload.get("topic"),
            tags=tags or [],
            score_hint=payload.get("score_hint"),
        )
        self._records.append(record)

        if self._client is None or vector is None:
            return

        if self.vector_size is None:
            self.vector_size = len(vector)
        elif self.vector_size != len(vector):
            logger.warning(
                "Vector dimension mismatch for %s (expected %s, received %s)",
                chunk_id,
                self.vector_size,
                len(vector),
            )
            return

        try:  # pragma: no cover - requires external service
            self._client.upsert(
                collection_name=self.collection,
                points=[
                    qmodels.PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to upsert vector %s to Qdrant: %s", chunk_id, exc)

    def export_records(self) -> List[Dict[str, Any]]:
        """Export records in a Fuse.js friendly structure."""

        return [
            {
                "id": record.id,
                "title": record.title,
                "content": record.content,
                "url": record.url,
                "topic": record.topic,
                "tags": record.tags,
                "score_hint": record.score_hint,
            }
            for record in self._records
        ]

