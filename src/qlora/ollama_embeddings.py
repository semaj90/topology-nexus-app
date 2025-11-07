"""Client utilities for interacting with Ollama's embedding endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import logging
import time

import requests


logger = logging.getLogger(__name__)


DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Raised when Ollama returns an error response."""


@dataclass
class EmbeddingResult:
    text: str
    vector: List[float]
    model: str
    elapsed_ms: float


class OllamaClient:
    """Minimal Ollama HTTP client focused on embedding workflows."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embed_batch(
        self,
        texts: Iterable[str],
        model: str = "embeddinggemma:latest",
        options: Optional[Dict[str, object]] = None,
    ) -> List[EmbeddingResult]:
        """Generate embeddings for ``texts`` using Ollama's REST API."""

        results: List[EmbeddingResult] = []
        for text in texts:
            text = text.strip()
            if not text:
                continue

            payload: Dict[str, object] = {"model": model, "prompt": text}
            if options:
                payload["options"] = options

            start_time = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/api/embeddings", json=payload, timeout=self.timeout
            )

            if response.status_code != 200:
                raise OllamaError(f"Ollama returned {response.status_code}: {response.text}")

            data = response.json()
            vector = data.get("embedding")
            if not isinstance(vector, list):
                raise OllamaError("Unexpected embedding payload returned by Ollama")

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            results.append(
                EmbeddingResult(
                    text=text,
                    vector=[float(value) for value in vector],
                    model=model,
                    elapsed_ms=elapsed_ms,
                )
            )

        logger.info("Generated %s embeddings with model %s", len(results), model)
        return results

