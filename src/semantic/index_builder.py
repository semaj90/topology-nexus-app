"""Utility helpers for generating Fuse.js friendly semantic indexes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import jsonlines

from .text_processor import SemanticProcessor
from .vector_registry import VectorRegistry
from .xrabbit_indexdb import XRabbitIndexDB

logger = logging.getLogger(__name__)


def _load_dataset(dataset_path: Path | str) -> List[Dict]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if path.suffix == ".jsonl":
        with jsonlines.open(path, "r") as reader:
            return list(reader)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
        if isinstance(data, list):
            return data
    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def build_semantic_index(
    dataset_path: Path | str,
    output_path: Path | str,
    qdrant_url: Optional[str] = None,
    collection: str = "topology_docs",
) -> int:
    """Generate a semantic index from a JSON/JSONL dataset."""

    dataset = _load_dataset(dataset_path)
    if not dataset:
        logger.warning("Dataset %s is empty", dataset_path)
        return 0

    processor = SemanticProcessor()
    registry = VectorRegistry(collection=collection, qdrant_url=qdrant_url)

    texts = [item.get("content", "") for item in dataset]
    embeddings = processor.compute_embeddings(texts)

    if embeddings.size:
        registry.ensure_collection(embeddings.shape[1])

    semantic_records: List[Dict] = []
    for idx, item in enumerate(dataset):
        summary = processor.extract_key_concepts(item.get("content", ""))
        chunk_id = item.get("chunk_id") or f"chunk-{idx}"
        payload = {
            "title": item.get("title") or item.get("metadata", {}).get("title"),
            "content": item.get("content", ""),
            "url": item.get("url") or item.get("metadata", {}).get("url"),
            "topic": item.get("topic") or item.get("metadata", {}).get("topic"),
            "score_hint": summary.get("concept_frequencies"),
            "metadata": item.get("metadata", {}),
        }

        vector = None
        if embeddings.size and idx < embeddings.shape[0]:
            vector = embeddings[idx].tolist()

        registry.register(chunk_id, vector, payload, summary.get("top_concepts"))

        semantic_records.append(
            {
                "id": chunk_id,
                "title": payload.get("title") or "Untitled",
                "content": payload.get("content", ""),
                "url": payload.get("url"),
                "topic": payload.get("topic"),
                "tags": summary.get("top_concepts", []),
                "score_hint": summary.get("concept_frequencies", {}),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(semantic_records, handle, indent=2, ensure_ascii=False)

    XRabbitIndexDB(output_path.with_suffix(".indexdb.json")).persist(semantic_records)
    logger.info("Semantic index with %s records written to %s", len(semantic_records), output_path)
    return len(semantic_records)

