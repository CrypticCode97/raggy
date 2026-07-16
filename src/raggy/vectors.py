from __future__ import annotations

import threading
from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, path: Path):
        self.path = path
        self.index = None
        self.dimension: int | None = None
        self.lock = threading.RLock()
        if path.exists():
            self.index = faiss.read_index(str(path))
            self.dimension = self.index.d

    def _create(self, dimension: int):
        base = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = 200
        base.hnsw.efSearch = 128
        self.index = faiss.IndexIDMap2(base)
        self.dimension = dimension

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if not ids:
            return
        with self.lock:
            if self.index is None:
                self._create(vectors.shape[1])
            if vectors.shape[1] != self.dimension:
                raise ValueError("Embedding dimension changed; rebuild the semantic index")
            self.index.add_with_ids(vectors.astype("float32"), np.asarray(ids, dtype="int64"))
            self.save()

    def search(self, query: np.ndarray, limit: int) -> list[tuple[int, float]]:
        with self.lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            scores, ids = self.index.search(query.astype("float32"), limit)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    def save(self) -> None:
        if self.index is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        faiss.write_index(self.index, str(tmp))
        tmp.replace(self.path)
