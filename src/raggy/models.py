from __future__ import annotations

import gc
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
    fallback_reason: str | None = None


class ModelOutOfMemoryError(RuntimeError):
    code = "gpu_out_of_memory"

    def __init__(self, device: str, operation: str):
        label = "GPU" if device == "cuda" else "Apple GPU" if device == "mps" else "system"
        super().__init__(
            f"{label} memory was exhausted while {operation}, even at batch size 1. "
            "Close other GPU-heavy applications or select the Balanced profile in Setup. "
            "Exact search remains available."
        )


class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._embedder = None
        self._reranker = None
        self._lock = threading.RLock()
        self.device = self._detect_device()
        self.error: str | None = None
        self.fallback_reason: str | None = None

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
        if self.settings.model_profile == "quality":
            return self.settings.embedding_model
        return self.settings.fallback_embedding_model

    @property
    def reranker_name(self) -> str:
        if self.settings.model_profile == "quality":
            return self.settings.reranker_model
        return self.settings.fallback_reranker_model

    def _switch_to_cpu(self, reason: str) -> None:
        if self.device == "cpu":
            return
        for model in (self._embedder, self._reranker):
            if model is None:
                continue
            try:
                model.to("cpu")
            except (AttributeError, RuntimeError):
                try:
                    model.model.to("cpu")
                except (AttributeError, RuntimeError):
                    pass
        self._release_memory()
        self.device = "cpu"
        self.fallback_reason = reason

    def _preflight_memory(self, operation: str) -> None:
        if self.device != "cuda":
            return
        try:
            import torch
            free_bytes, _ = torch.cuda.mem_get_info()
            required_gib = 3.0 if self.settings.model_profile == "quality" else 1.5
            if free_bytes < required_gib * 1024**3:
                self._switch_to_cpu(
                    f"Using CPU for {operation}: only {free_bytes / 1024**3:.1f} GB of GPU "
                    f"memory is free; approximately {required_gib:.1f} GB is required."
                )
        except (ImportError, RuntimeError):
            pass

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        message = str(exc).casefold()
        if "out of memory" in message or "not enough memory" in message:
            return True
        try:
            import torch
            return isinstance(exc, torch.OutOfMemoryError)
        except (ImportError, AttributeError):
            return False

    def _release_memory(self) -> None:
        gc.collect()
        try:
            import torch
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            elif self.device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def load_embedder(self) -> None:
        with self._lock:
            if self._embedder is not None:
                return
            self._preflight_memory("embeddings")
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    self.embedding_name, device=self.device, local_files_only=True
                )
                self.error = None
            except Exception as exc:
                self._embedder = None
                self._release_memory()
                if self._is_oom(exc):
                    if self.device != "cpu":
                        self._switch_to_cpu(
                            "The embedding model did not fit in GPU memory; using CPU instead."
                        )
                        return self.load_embedder()
                    error = ModelOutOfMemoryError(self.device, "loading the embedding model")
                    self.error = str(error)
                    raise error from exc
                self.error = str(exc)
                raise RuntimeError(f"Local embedding model initialization failed: {exc}") from exc

    def load_reranker(self) -> None:
        with self._lock:
            if self._reranker is not None:
                return
            self._preflight_memory("reranking")
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(
                    self.reranker_name, device=self.device, local_files_only=True
                )
                self.error = None
            except Exception as exc:
                self._reranker = None
                self._release_memory()
                if self._is_oom(exc):
                    if self.device != "cpu":
                        self._switch_to_cpu(
                            "The reranker did not fit in GPU memory; using CPU instead."
                        )
                        return self.load_reranker()
                    error = ModelOutOfMemoryError(self.device, "loading the reranker")
                    self.error = str(error)
                    raise error from exc
                self.error = str(exc)
                raise RuntimeError(f"Local reranker initialization failed: {exc}") from exc

    def load(self) -> None:
        self.load_embedder()
        self.load_reranker()

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        self.load_embedder()
        prefix = "query: " if query and "e5" in self.embedding_name else "passage: " if "e5" in self.embedding_name else ""
        prepared = [prefix + text for text in texts]
        batch_size = min(32, max(1, len(prepared)))
        while batch_size >= 1:
            try:
                with self._lock:
                    result = self._embedder.encode(
                        prepared, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                self.error = None
                return np.asarray(result, dtype="float32")
            except Exception as exc:
                if not self._is_oom(exc):
                    raise
                self._release_memory()
                if batch_size == 1:
                    if self.device != "cpu":
                        self._switch_to_cpu(
                            "GPU memory was exhausted while embedding; continuing on CPU."
                        )
                        continue
                    error = ModelOutOfMemoryError(self.device, "creating embeddings")
                    self.error = str(error)
                    raise error from exc
                batch_size = max(1, batch_size // 2)
        raise AssertionError("unreachable")

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        self.load_reranker()
        pairs = [(query, text) for text in texts]
        batch_size = min(16, max(1, len(pairs)))
        while batch_size >= 1:
            try:
                with self._lock:
                    scores = self._reranker.predict(
                        pairs, batch_size=batch_size, show_progress_bar=False
                    )
                self.error = None
                return [float(score) for score in scores]
            except Exception as exc:
                if not self._is_oom(exc):
                    raise
                self._release_memory()
                if batch_size == 1:
                    if self.device != "cpu":
                        self._switch_to_cpu(
                            "GPU memory was exhausted while reranking; continuing on CPU."
                        )
                        continue
                    error = ModelOutOfMemoryError(self.device, "reranking evidence")
                    self.error = str(error)
                    raise error from exc
                batch_size = max(1, batch_size // 2)
        raise AssertionError("unreachable")

    def status(self) -> ModelStatus:
        return ModelStatus(
            profile=self.settings.model_profile,
            device=self.device,
            embedding_model=self.embedding_name,
            reranker_model=self.reranker_name,
            loaded=self._embedder is not None,
            error=self.error,
            fallback_reason=self.fallback_reason,
        )
