from __future__ import annotations

import json
import sys
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGGY_", extra="ignore")

    data_dir: Path = Path(user_data_dir("raggy", appauthor=False))
    cache_dir: Path = Path(user_cache_dir("raggy", appauthor=False))
    model_profile: str = "quality"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    fallback_embedding_model: str = "intfloat/multilingual-e5-small"
    fallback_reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ocr_languages: str = "eng"
    tessdata: Path | None = None
    chunk_tokens: int = 350
    chunk_overlap: int = 50
    parser_workers: int = 2
    host: str = "127.0.0.1"
    port: int = 7734

    @property
    def db_path(self) -> Path:
        return self.data_dir / "raggy.sqlite3"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "vectors.faiss"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir / "runtime"

    @property
    def static_dir(self) -> Path:
        return Path(__file__).parent / "static"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def activate_runtime(self) -> None:
        """Make wizard-managed packages importable without tying them to .venv."""
        if self.runtime_dir.exists() and str(self.runtime_dir) not in sys.path:
            sys.path.insert(0, str(self.runtime_dir))

    def apply_setup_file(self) -> None:
        path = self.data_dir / "setup.json"
        if not path.exists():
            return
        values = json.loads(path.read_text())
        self.model_profile = values.get("model_profile", self.model_profile)
        self.ocr_languages = values.get("ocr_languages", self.ocr_languages)
        if values.get("tessdata"):
            self.tessdata = Path(values["tessdata"])
