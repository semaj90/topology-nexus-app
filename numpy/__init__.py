"""Extremely small subset of numpy for offline testing."""

from __future__ import annotations

from typing import Iterable, List, Sequence


class SimpleArray(list):
    def __init__(self, data: Iterable):
        super().__init__(data)

    @property
    def shape(self):
        if not self:
            return (0,)
        first = self[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            return (len(self), len(first))
        return (len(self),)

    @property
    def size(self) -> int:
        if not self:
            return 0
        if isinstance(self[0], Sequence) and not isinstance(self[0], (str, bytes)):
            return sum(len(row) if isinstance(row, Sequence) else 1 for row in self)
        return len(self)

    def tolist(self):
        return list(self)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if isinstance(value, list) and not isinstance(value, SimpleArray):
            return SimpleArray(value)
        return value


def array(data: Iterable) -> SimpleArray:
    if isinstance(data, SimpleArray):
        return data
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return SimpleArray([array(item) if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) else item for item in data])
    return SimpleArray([data])


def dot(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> SimpleArray:
    results: List[float] = []
    for row in matrix:
        results.append(sum(float(a) * float(b) for a, b in zip(row, vector)))
    return SimpleArray(results)


def argsort(sequence: Sequence[float]):
    return SimpleArray(sorted(range(len(sequence)), key=lambda idx: sequence[idx]))


def zeros(shape):
    if isinstance(shape, int):
        return SimpleArray([0.0] * shape)
    rows, cols = shape
    return SimpleArray([[0.0] * cols for _ in range(rows)])
