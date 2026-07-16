from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import threading
import urllib.request
from dataclasses import dataclass

from huggingface_hub import snapshot_download
import pymupdf

from .config import Settings
from .db import Database
from .models import ModelManager


OCR_LANGUAGES = [
    {"code": "eng", "name": "English", "download_mb": 4.1},
    {"code": "deu", "name": "German", "download_mb": 4.2},
    {"code": "fra", "name": "French", "download_mb": 3.9},
    {"code": "spa", "name": "Spanish", "download_mb": 2.2},
    {"code": "ita", "name": "Italian", "download_mb": 4.8},
    {"code": "nld", "name": "Dutch", "download_mb": 4.0},
    {"code": "pol", "name": "Polish", "download_mb": 4.6},
    {"code": "por", "name": "Portuguese", "download_mb": 2.1},
    {"code": "swe", "name": "Swedish", "download_mb": 4.0},
    {"code": "ces", "name": "Czech", "download_mb": 4.0},
]


@dataclass(slots=True)
class SetupState:
    state: str = "idle"
    stage: str = ""
    detail: str = ""
    completed: int = 0
    total: int = 0
    error: str | None = None
    restart_required: bool = False
    configured: bool = False
    def public(self) -> dict:
        return {
            "state": self.state, "stage": self.stage, "detail": self.detail,
            "completed": self.completed, "total": self.total, "error": self.error,
            "restart_required": self.restart_required, "configured": self.configured,
        }


class SetupCoordinator:
    def __init__(self, settings: Settings, models: ModelManager, db: Database):
        self.settings = settings
        self.models = models
        self.db = db
        self.settings.activate_runtime()
        self.status = SetupState(configured=(settings.data_dir / "setup.json").exists())
        self._worker: threading.Thread | None = None
        self.on_complete = None
        self._reconcile_ocr_failures()

    def _reconcile_ocr_failures(self) -> None:
        languages = self.settings.ocr_languages.split("+")
        if not self.status.configured or not self.settings.tessdata:
            return
        if any(not (self.settings.tessdata / f"{language}.traineddata").is_file() for language in languages):
            return
        try:
            self._verify_ocr(self.settings.tessdata, languages)
        except RuntimeError:
            return
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE documents SET state='pending' WHERE id IN "
                "(SELECT DISTINCT document_id FROM pages WHERE warning LIKE 'OCR unavailable:%')"
            )

    def capabilities(self) -> dict:
        runtime, runtime_error = self._runtime_status()
        installed_ocr = self._installed_ocr_languages()
        with self.db.connect() as conn:
            retry_count = conn.execute(
                "SELECT COUNT(DISTINCT document_id) FROM pages WHERE warning LIKE 'OCR unavailable:%'"
            ).fetchone()[0]
        return {
            "configured": self.status.configured,
            "device": self.models.device,
            "gpu_name": self._gpu_name(),
            "semantic_runtime": runtime,
            "semantic_runtime_error": runtime_error,
            "ocr_ready": bool(self.settings.tessdata and self.settings.tessdata.is_dir()),
            "installed_ocr_languages": installed_ocr,
            "ocr_retry_documents": retry_count,
            "profiles": [
                {"id": "quality", "name": "Maximum quality", "description": "BGE-M3 embedding and reranker; best on a GPU.", "download_gb": 4.6},
                {"id": "fallback", "name": "Balanced", "description": "Smaller multilingual models for lower memory use.", "download_gb": 1.1},
                {"id": "exact", "name": "Exact search only", "description": "Keyword and phrase search without ML models.", "download_gb": 0},
            ],
            "ocr_languages": OCR_LANGUAGES,
        }

    def _installed_ocr_languages(self) -> list[str]:
        if not self.settings.tessdata or not self.settings.tessdata.is_dir():
            return []
        return sorted(path.stem for path in self.settings.tessdata.glob("*.traineddata"))

    @staticmethod
    def _verify_ocr(tessdata, languages: list[str]) -> None:
        document = pymupdf.open()
        try:
            page = document.new_page()
            page.insert_text((72, 72), "Raggy OCR readiness check")
            page.get_textpage_ocr(
                language="+".join(languages), dpi=72, full=True, tessdata=str(tessdata)
            )
        except Exception as exc:
            raise RuntimeError(f"OCR validation failed: {exc}") from exc
        finally:
            document.close()

    def _runtime_status(self) -> tuple[bool, str | None]:
        self.settings.activate_runtime()
        if importlib.util.find_spec("sentence_transformers") is None:
            return False, "Semantic runtime is not installed"
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: F401
            return True, None
        except Exception as exc:
            return False, str(exc)

    @staticmethod
    def _gpu_name() -> str | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3, check=False,
            )
            return result.stdout.strip().splitlines()[0] if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def start(self, profile: str, languages: list[str], install_runtime: bool) -> dict:
        if self._worker and self._worker.is_alive():
            raise RuntimeError("Setup is already running")
        if profile not in {"quality", "fallback", "exact"}:
            raise ValueError("Unknown model profile")
        if not languages or any(not re.fullmatch(r"[a-z]{3}(?:-[a-z]+)?", item) for item in languages):
            raise ValueError("Select at least one valid OCR language")
        self.status = SetupState(state="running", stage="Preparing", total=len(languages) + (2 if profile != "exact" else 0), configured=self.status.configured)
        self._worker = threading.Thread(target=self._run, args=(profile, languages, install_runtime), daemon=True)
        self._worker.start()
        return self.status.public()

    def _update(self, stage: str, detail: str = "") -> None:
        self.status.stage = stage
        self.status.detail = detail

    def _run(self, profile: str, languages: list[str], install_runtime: bool) -> None:
        try:
            runtime_available = self._runtime_status()[0]
            durable_runtime = (self.settings.runtime_dir / "sentence_transformers").is_dir()
            runtime_missing = not runtime_available or not durable_runtime
            if profile != "exact" and runtime_missing and install_runtime:
                self.status.total += 1
                self._update("Installing semantic runtime", "This can take several minutes on CUDA systems.")
                uv = shutil.which("uv")
                if not uv:
                    raise RuntimeError("The semantic runtime is missing and uv is not available to install it")
                subprocess.run(
                    [
                        uv, "pip", "install", "--target", str(self.settings.runtime_dir),
                        "sentence-transformers>=5,<6", "transformers>=4.57,<5",
                        "huggingface-hub>=0.33,<1",
                    ],
                    check=True,
                )
                importlib.invalidate_caches()
                self.settings.activate_runtime()
                self.status.completed += 1
                self.status.restart_required = False
            if profile != "exact":
                names = (
                    [self.settings.embedding_model, self.settings.reranker_model]
                    if profile == "quality"
                    else [self.settings.fallback_embedding_model, self.settings.fallback_reranker_model]
                )
                for name in names:
                    self._update("Downloading local models", name)
                    snapshot_download(name)
                    self.status.completed += 1
            tessdata = self.settings.data_dir / "tessdata"
            tessdata.mkdir(parents=True, exist_ok=True)
            for language in languages:
                self._update("Installing OCR languages", language)
                destination = tessdata / f"{language}.traineddata"
                if not destination.exists():
                    urllib.request.urlretrieve(
                        f"https://github.com/tesseract-ocr/tessdata_fast/raw/main/{language}.traineddata",
                        destination,
                    )
                self.status.completed += 1
            self._update("Validating OCR", "+".join(languages))
            self.status.total += 1
            self._verify_ocr(tessdata, languages)
            self.status.completed += 1
            config = {
                "model_profile": profile,
                "ocr_languages": "+".join(languages),
                "tessdata": str(tessdata),
            }
            (self.settings.data_dir / "setup.json").write_text(json.dumps(config, indent=2))
            self.settings.model_profile = profile
            self.settings.ocr_languages = config["ocr_languages"]
            self.settings.tessdata = tessdata
            self.models.device = self.models._detect_device()
            with self.db.transaction() as conn:
                conn.execute(
                    "UPDATE documents SET state='pending' WHERE id IN "
                    "(SELECT DISTINCT document_id FROM pages WHERE warning LIKE 'OCR unavailable:%')"
                )
            self.status.state = "complete"
            self.status.stage = "Ready"
            self.status.detail = "Offline assets and OCR are validated. Rescan to retry earlier OCR failures."
            self.status.configured = True
            if self.on_complete and profile != "exact":
                threading.Thread(target=self.on_complete, daemon=True).start()
        except Exception as exc:
            self.status.state = "failed"
            self.status.error = str(exc)
            self.status.stage = "Setup failed"
