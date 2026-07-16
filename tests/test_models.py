import numpy as np
import pytest

from raggy.config import Settings
from raggy.models import ModelManager, ModelOutOfMemoryError


class AdaptiveEmbedder:
    def __init__(self, succeeds_at: int | None):
        self.succeeds_at = succeeds_at
        self.batch_sizes: list[int] = []

    def encode(self, texts, *, batch_size, **kwargs):
        self.batch_sizes.append(batch_size)
        if self.succeeds_at is None or batch_size > self.succeeds_at:
            raise RuntimeError("CUDA out of memory")
        return np.ones((len(texts), 4), dtype="float32")


def manager_with(embedder: AdaptiveEmbedder) -> ModelManager:
    manager = ModelManager(Settings())
    manager.device = "cuda"
    manager._embedder = embedder
    return manager


def test_embedding_retries_with_smaller_batches(monkeypatch):
    embedder = AdaptiveEmbedder(succeeds_at=2)
    manager = manager_with(embedder)
    monkeypatch.setattr(manager, "_release_memory", lambda: None)
    result = manager.encode(["evidence"] * 8)
    assert result.shape == (8, 4)
    assert embedder.batch_sizes == [8, 4, 2]


def test_embedding_reports_actionable_oom(monkeypatch):
    embedder = AdaptiveEmbedder(succeeds_at=None)
    manager = manager_with(embedder)
    monkeypatch.setattr(manager, "_release_memory", lambda: None)
    with pytest.raises(ModelOutOfMemoryError, match="Balanced profile") as error:
        manager.encode(["evidence"] * 4)
    assert error.value.code == "gpu_out_of_memory"
    assert embedder.batch_sizes == [4, 2, 1, 1]
    assert manager.device == "cpu"
    assert manager.fallback_reason is not None
