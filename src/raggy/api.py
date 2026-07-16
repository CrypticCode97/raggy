from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .db import Database
from .indexer import IndexCoordinator
from .models import ModelManager, ModelOutOfMemoryError
from .search import SearchService
from .setup_service import SetupCoordinator
from .vectors import VectorIndex


class SourceCreate(BaseModel):
    path: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: Literal["hybrid", "exact", "semantic"] = "hybrid"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)


class SetupRequest(BaseModel):
    profile: Literal["quality", "fallback", "exact"]
    languages: list[str]
    install_runtime: bool = True


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    settings.apply_setup_file()
    settings.activate_runtime()
    db = Database(settings.db_path)
    db.initialize()
    models = ModelManager(settings)
    vectors = VectorIndex(settings.index_path)
    indexer = IndexCoordinator(db, settings, models, vectors)
    search = SearchService(db, models, vectors)
    setup = SetupCoordinator(settings, models, db)
    setup.on_complete = indexer.backfill_missing

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        repair_task = None
        if setup.status.configured and settings.model_profile != "exact":
            repair_task = asyncio.create_task(asyncio.to_thread(indexer.backfill_missing))
        yield
        indexer.cancel()
        if repair_task and repair_task.done():
            repair_task.exception()

    app = FastAPI(title="Raggy Evidence Search", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.db = db
    app.state.models = models
    app.state.indexer = indexer
    app.state.search = search
    app.state.setup = setup

    @app.get("/api/health")
    def health():
        return {"status": "ok", "offline": True}

    @app.get("/api/status")
    def status():
        model_status = models.status()
        with db.connect() as conn:
            counts = conn.execute("SELECT COUNT(*) documents, COALESCE(SUM(page_count),0) pages FROM documents WHERE active=1").fetchone()
            semantic = conn.execute(
                "SELECT COUNT(*) total, COUNT(embedding_id) embedded FROM passages WHERE active=1"
            ).fetchone()
            job = conn.execute("SELECT * FROM index_jobs ORDER BY id DESC LIMIT 1").fetchone()
        semantic_status = indexer.semantic_status()
        semantic_status.update({"total": semantic["total"], "embedded": semantic["embedded"], "ready": semantic["total"] > 0 and semantic["total"] == semantic["embedded"]})
        return {"models": model_status.__dict__ if hasattr(model_status, "__dict__") else {field: getattr(model_status, field) for field in model_status.__slots__}, "documents": counts["documents"], "pages": counts["pages"], "semantic": semantic_status, "job": dict(job) if job else None, "ocr_languages": settings.ocr_languages}

    @app.get("/api/setup")
    def setup_status():
        return {"capabilities": setup.capabilities(), "job": setup.status.public()}

    @app.post("/api/setup", status_code=202)
    def start_setup(payload: SetupRequest):
        try:
            return setup.start(payload.profile, payload.languages, payload.install_runtime)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/sources")
    def list_sources():
        with db.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM source_roots ORDER BY path")]

    @app.post("/api/sources", status_code=201)
    def add_source(payload: SourceCreate):
        path = Path(payload.path).expanduser().resolve()
        if not path.is_dir():
            raise HTTPException(400, "Source must be an existing directory")
        with db.transaction() as conn:
            try:
                cursor = conn.execute("INSERT INTO source_roots(path) VALUES (?)", (str(path),))
            except Exception as exc:
                raise HTTPException(409, "Source already exists") from exc
        return {"id": cursor.lastrowid, "path": str(path)}

    @app.delete("/api/sources/{source_id}", status_code=204)
    def delete_source(source_id: int):
        with db.transaction() as conn:
            cursor = conn.execute("DELETE FROM source_roots WHERE id=?", (source_id,))
            if not cursor.rowcount:
                raise HTTPException(404, "Source not found")

    @app.post("/api/index-jobs", status_code=202)
    async def create_index_job():
        with db.connect() as conn:
            if not conn.execute("SELECT 1 FROM source_roots LIMIT 1").fetchone():
                raise HTTPException(400, "Add a source folder first")
        job_id = indexer.create_job()
        await indexer.start(job_id)
        return {"id": job_id, "state": "queued"}

    @app.get("/api/index-jobs/{job_id}")
    def get_index_job(job_id: int):
        with db.connect() as conn:
            row = conn.execute("SELECT * FROM index_jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job not found")
        return dict(row)

    @app.delete("/api/index-jobs/{job_id}", status_code=202)
    def cancel_index_job(job_id: int):
        indexer.cancel()
        return {"id": job_id, "state": "cancelling"}

    @app.post("/api/search")
    async def run_search(payload: SearchRequest):
        try:
            output = await asyncio.to_thread(search.search, payload.query, payload.mode, payload.page, payload.page_size)
        except ModelOutOfMemoryError as exc:
            raise HTTPException(
                503,
                {"code": exc.code, "message": str(exc), "recoverable": True},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"query": payload.query, "mode": payload.mode, "page": payload.page, "page_size": payload.page_size, "total": output.total, "took_ms": round(output.took_ms, 2), "warning": output.warning, "results": output.results}

    @app.get("/api/documents/{document_id}/pdf")
    def pdf(document_id: int):
        with db.connect() as conn:
            row = conn.execute("SELECT path FROM documents WHERE id=? AND active=1", (document_id,)).fetchone()
        if not row or not Path(row["path"]).is_file():
            raise HTTPException(404, "Original PDF is unavailable")
        return FileResponse(row["path"], media_type="application/pdf", filename=Path(row["path"]).name)

    @app.get("/api/documents")
    def documents(limit: int = Query(100, ge=1, le=1000)):
        with db.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT id,title,path,page_count,state,error,indexed_at FROM documents WHERE active=1 ORDER BY title LIMIT ?", (limit,))]

    static = settings.static_dir
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="frontend")
    return app


app = create_app()
