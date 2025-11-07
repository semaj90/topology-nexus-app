"""IndexedDB-inspired persistence layer with XRabbit signalling placeholders."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


class XRabbitIndexDB:
    """Persists semantic memory to disk and mimics XRabbit queue notifications."""

    def __init__(self, storage_path: Path | str = Path("data/indexdb_memory.json")) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def persist(self, records: Iterable[Dict[str, Any]]) -> None:
        data = list(records)
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

        logger.info("Persisted %s records to %s", len(data), self.storage_path)
        self._notify_queue("persist", len(data))

    def read(self) -> List[Dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        try:
            with self.storage_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            logger.error("Failed to read XRabbit IndexDB store: %s", exc)
            return []

    def _notify_queue(self, event: str, count: int) -> None:
        """Placeholder for the future Go + NATS QUIC microservice integration."""

        logger.debug("XRabbit placeholder event '%s' with %s records", event, count)

