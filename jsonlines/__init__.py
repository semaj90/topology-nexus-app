"""Lightweight jsonlines compatibility shim for offline test environments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional, TextIO


class _Writer:
    def __init__(self, handle: TextIO) -> None:
        self._handle = handle

    def write(self, data) -> None:
        json.dump(data, self._handle, ensure_ascii=False)
        self._handle.write("\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _Reader(Iterator):
    def __init__(self, handle: TextIO) -> None:
        self._handle = handle

    def __iter__(self) -> "_Reader":
        return self

    def __next__(self):
        line = self._handle.readline()
        if not line:
            raise StopIteration
        return json.loads(line)

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_Reader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class _JsonLinesContext:
    def __init__(self, path: Path, mode: str) -> None:
        self._path = path
        self._mode = mode
        self._handle: Optional[TextIO] = None

    def __enter__(self):
        self._handle = self._path.open(self._mode, encoding="utf-8")
        if "r" in self._mode:
            return _Reader(self._handle)
        return _Writer(self._handle)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle:
            self._handle.close()


def open(path: str | Path, mode: str = "r"):
    return _JsonLinesContext(Path(path), mode)


def Writer(handle: TextIO) -> _Writer:
    return _Writer(handle)


def Reader(handle: TextIO) -> _Reader:
    return _Reader(handle)


__all__ = ["open", "Writer", "Reader"]
