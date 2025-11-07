"""Utilities for building QLoRA-ready datasets from heterogeneous sources.

This module focuses on turning raw user-provided documents into structured
datasets that can be consumed by downstream training and retrieval pipelines.
The implementation intentionally avoids heavyweight framework assumptions so it
can run in constrained desktop environments (such as a Tauri runtime launched
inside WSL2) while still producing high-quality JSONL corpora.

The core entry point is :class:`DatasetBuilder`, which accepts a collection of
paths, normalises the textual content, chunks it with optional overlap, and
emits JSONL lines that include lightweight metadata.  The module also exposes a
small helper for shipping data directly into Ollama's embedding API so the same
dataset can be indexed for retrieval-augmented generation (RAG) workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Sequence
import html
import json
import logging
import re

import jsonlines
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".json", ".jsonl"}


@dataclass
class DocumentChunk:
    """Represents a single chunk of processed text."""

    id: str
    source: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


class DatasetBuilder:
    """Build JSONL datasets from a collection of documents.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.  Chunking happens on paragraph
        boundaries whenever possible.
    overlap:
        Number of characters to overlap between chunks.  This helps preserve
        context across chunk boundaries and is particularly useful for
        retrieval-augmented workflows.
    """

    def __init__(self, chunk_size: int = 1_024, overlap: int = 200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def build_from_paths(self, paths: Sequence[Path]) -> List[DocumentChunk]:
        """Load documents from ``paths`` and return processed chunks."""

        chunks: List[DocumentChunk] = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                logger.warning("Skipping missing document %s", path)
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning("Skipping unsupported extension for %s", path)
                continue

            logger.debug("Loading document from %s", path)
            text = self._load_document(path)
            if not text.strip():
                logger.info("Skipping empty document %s", path)
                continue

            document_chunks = list(self._chunk_text(text))
            for index, chunk_text in enumerate(document_chunks):
                chunk_id = f"{path.stem}-{index:04d}"
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        source=str(path),
                        text=chunk_text,
                        metadata={"chunk_index": str(index)},
                    )
                )

        logger.info("Prepared %s chunks from %s documents", len(chunks), len(paths))
        return chunks

    def write_jsonl(self, chunks: Sequence[DocumentChunk], output_path: Path) -> Path:
        """Persist ``chunks`` to ``output_path`` in JSONL format."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(output_path, mode="w") as writer:
            for chunk in chunks:
                writer.write(
                    {
                        "id": chunk.id,
                        "source": chunk.source,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }
                )

        logger.info("Wrote dataset with %s chunks to %s", len(chunks), output_path)
        return output_path

    def build_jsonl(self, paths: Sequence[Path], output_path: Path) -> Path:
        """Convenience helper that ingests documents and writes JSONL output."""

        chunks = self.build_from_paths(paths)
        return self.write_jsonl(chunks, output_path)

    def _load_document(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".markdown"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix in {".html", ".htm"}:
            return self._extract_html(path.read_text(encoding="utf-8", errors="ignore"))
        if suffix == ".jsonl":
            return self._read_jsonl_as_text(path)
        if suffix == ".json":
            return self._read_json_as_text(path)
        raise ValueError(f"Unsupported document type for {path}")

    def _chunk_text(self, text: str) -> Iterator[str]:
        cleaned = re.sub(r"\s+", " ", text.strip())
        if not cleaned:
            return iter(())

        paragraphs = [paragraph.strip() for paragraph in cleaned.split("\n") if paragraph.strip()]
        buffer: List[str] = []
        current_length = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if current_length + len(paragraph) + 1 <= self.chunk_size:
                buffer.append(paragraph)
                current_length += len(paragraph) + 1
                continue

            if buffer:
                yield "\n\n".join(buffer)

            # Start a new buffer while preserving overlap
            buffer_text = "\n\n".join(buffer)
            if buffer_text:
                overlap_text = buffer_text[-self.overlap :]
            else:
                overlap_text = ""

            buffer = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_length = sum(len(part) for part in buffer) + (2 * (len(buffer) - 1))

        if buffer:
            yield "\n\n".join(part for part in buffer if part)

    def _extract_html(self, html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        text = soup.get_text(separator="\n")
        return html.unescape(text)

    def _read_json_as_text(self, path: Path) -> str:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _read_jsonl_as_text(self, path: Path) -> str:
        lines: List[str] = []
        with jsonlines.open(path, mode="r") as reader:
            for entry in reader:
                lines.append(json.dumps(entry, ensure_ascii=False))
        return "\n".join(lines)


def chunk_documents(paths: Iterable[str], output_path: str, chunk_size: int = 1_024, overlap: int = 200) -> Path:
    """High-level helper for scripting contexts."""

    builder = DatasetBuilder(chunk_size=chunk_size, overlap=overlap)
    resolved_paths = [Path(path) for path in paths]
    return builder.build_jsonl(resolved_paths, Path(output_path))

