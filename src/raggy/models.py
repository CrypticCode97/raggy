from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .config import Settings


@dataclass(slots=True)
class ModelStatus:
    profile: str
    device: str
    embedding_model: str
    reranker_model: str
    loaded: bool
    error: str | None = None


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._embedder = None
        self._reranker = None
        self._lock = threading.RLock()
        self.device = self._detect_device()
        self.error: str | None = None

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @property
    def embedding_name(self) -> str:
        if self.settings.model_profile == "quality" and self.device != "cpu":
            return self.settings.embedding_model
        return self.settings.fallback_embedding_model

    @property
    def reranker_name(self) -> str:
        if self.settings.model_profile == "quality" and self.device != "cpu":
            return self.settings.reranker_model
        return self.settings.fallback_reranker_model

    def load(self) -> None:
        with self._lock:
            if self._embedder is not None:
                return
            try:
                from sentence_transformers import CrossEncoder, SentenceTransformer
                self._embedder = SentenceTransformer(
                    self.embedding_name, device=self.device, local_files_only=True
                )
                self._reranker = CrossEncoder(
                    self.reranker_name, device=self.device, local_files_only=True
                )
            except Exception as exc:
                self.error = str(exc)
                self._embedder = None
                self._reranker = None
                raise RuntimeError(
                    f"Local model initialization failed: {exc}"
                ) from exc

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        self.load()
        prefix = "query: " if query and "e5" in self.embedding_name else "passage: " if "e5" in self.embedding_name else ""
        prepared = [prefix + text for text in texts]
        with self._lock:
            return np.asarray(self._embedder.encode(
                prepared, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            ), dtype="float32")

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.load()
        with self._lock:
            scores = self._reranker.predict([(query, text) for text in texts], show_progress_bar=False)
        return [float(score) for score in scores]

    def status(self) -> ModelStatus:
        return ModelStatus(
            profile=self.settings.model_profile,
            device=self.device,
            embedding_model=self.embedding_name,
            reranker_model=self.reranker_name,
            loaded=self._embedder is not None,
            error=self.error,
        )
