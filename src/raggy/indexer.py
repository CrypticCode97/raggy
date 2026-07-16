from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from .config import Settings
from .db import Database
from .extract import chunk_words, extract_pdf, sha256_file
from .models import ModelManager
from .vectors import VectorIndex


class IndexCoordinator:
    def __init__(self, db: Database, settings: Settings, models: ModelManager, vectors: VectorIndex):
        self.db = db
        self.settings = settings
        self.models = models
        self.vectors = vectors
        self._running = threading.Lock()
        self._cancel = threading.Event()

    def create_job(self) -> int:
        with self.db.transaction() as conn:
            cursor = conn.execute("INSERT INTO index_jobs(state) VALUES ('queued')")
            return int(cursor.lastrowid)

    async def start(self, job_id: int) -> None:
        asyncio.create_task(asyncio.to_thread(self._run, job_id))

    def cancel(self) -> None:
        self._cancel.set()

    def _discover(self) -> list[Path]:
        with self.db.connect() as conn:
            roots = [Path(row["path"]) for row in conn.execute("SELECT path FROM source_roots")]
        found: dict[str, Path] = {}
        for root in roots:
            if root.is_dir():
                for path in root.rglob("*.pdf"):
                    if path.is_file():
                        found[str(path.resolve())] = path.resolve()
        return sorted(found.values())

    def _run(self, job_id: int) -> None:
        if not self._running.acquire(blocking=False):
            with self.db.transaction() as conn:
                conn.execute("UPDATE index_jobs SET state='failed', error=? WHERE id=?", ("Another indexing job is running", job_id))
            return
        self._cancel.clear()
        try:
            paths = self._discover()
            with self.db.transaction() as conn:
                conn.execute("UPDATE index_jobs SET state='running', total_files=? WHERE id=?", (len(paths), job_id))
            discovered = {str(path) for path in paths}
            with self.db.transaction() as conn:
                rows = conn.execute("SELECT id, path FROM documents WHERE active=1").fetchall()
                for row in rows:
                    if row["path"] not in discovered:
                        conn.execute("UPDATE documents SET active=0 WHERE id=?", (row["id"],))
                        conn.execute("UPDATE passages SET active=0 WHERE page_id IN (SELECT id FROM pages WHERE document_id=?)", (row["id"],))
            for path in paths:
                if self._cancel.is_set():
                    with self.db.transaction() as conn:
                        conn.execute("UPDATE index_jobs SET state='cancelled', finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
                    return
                self._process_one(path, job_id)
            with self.db.transaction() as conn:
                conn.execute("UPDATE index_jobs SET state='complete', finished_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
                conn.execute("UPDATE source_roots SET last_scanned_at=CURRENT_TIMESTAMP")
        except Exception as exc:
            with self.db.transaction() as conn:
                conn.execute("UPDATE index_jobs SET state='failed', error=?, finished_at=CURRENT_TIMESTAMP WHERE id=?", (str(exc), job_id))
        finally:
            self._running.release()

    def _process_one(self, path: Path, job_id: int) -> None:
        stat = path.stat()
        with self.db.connect() as conn:
            existing = conn.execute("SELECT * FROM documents WHERE path=?", (str(path),)).fetchone()
        if existing and existing["size"] == stat.st_size and existing["mtime_ns"] == stat.st_mtime_ns and existing["state"] == "indexed":
            with self.db.connect() as conn:
                missing = conn.execute(
                    "SELECT COUNT(*) count FROM passages WHERE active=1 AND embedding_id IS NULL "
                    "AND page_id IN (SELECT id FROM pages WHERE document_id=?)",
                    (existing["id"],),
                ).fetchone()["count"]
            if missing:
                self._backfill_document(int(existing["id"]))
            with self.db.transaction() as conn:
                conn.execute("UPDATE documents SET active=1 WHERE id=?", (existing["id"],))
                conn.execute(
                    "UPDATE index_jobs SET processed_files=processed_files+1, "
                    "indexed_files=indexed_files+?, current_path=? WHERE id=?",
                    (1 if missing else 0, str(path), job_id),
                )
            return
        try:
            digest = sha256_file(path)
            pages = extract_pdf(path, self.settings.ocr_languages, self.settings.tessdata)
            with self.db.transaction() as conn:
                if existing:
                    doc_id = int(existing["id"])
                    conn.execute("DELETE FROM pages WHERE document_id=?", (doc_id,))
                    conn.execute("UPDATE documents SET title=?, size=?, mtime_ns=?, sha256=?, page_count=?, state='indexing', error=NULL, active=1 WHERE id=?", (path.stem, stat.st_size, stat.st_mtime_ns, digest, len(pages), doc_id))
                else:
                    cursor = conn.execute("INSERT INTO documents(path,title,size,mtime_ns,sha256,page_count,state) VALUES (?,?,?,?,?,?,'indexing')", (str(path), path.stem, stat.st_size, stat.st_mtime_ns, digest, len(pages)))
                    doc_id = int(cursor.lastrowid)
                passages: list[tuple[int, str]] = []
                for page in pages:
                    cursor = conn.execute("INSERT INTO pages(document_id,page_number,text,normalized_text,width,height,extraction_method,layout_json,warning) VALUES (?,?,?,?,?,?,?,?,?)", (doc_id, page.page_number, page.text, page.normalized_text, page.width, page.height, page.method, page.layout, page.warning))
                    page_id = int(cursor.lastrowid)
                    for ordinal, (text, word_start, word_end) in enumerate(chunk_words(page.layout)):
                        pcur = conn.execute("INSERT INTO passages(page_id,ordinal,text,word_start,word_end) VALUES (?,?,?,?,?)", (page_id, ordinal, text, word_start, word_end))
                        passages.append((int(pcur.lastrowid), text))
            semantic_error = None
            if passages:
                try:
                    embeddings = self.models.encode([text for _, text in passages])
                    ids = [pid for pid, _ in passages]
                    self.vectors.add(ids, embeddings)
                    with self.db.transaction() as conn:
                        conn.executemany("UPDATE passages SET embedding_id=? WHERE id=?", [(pid, pid) for pid in ids])
                except RuntimeError as exc:
                    semantic_error = str(exc)
            with self.db.transaction() as conn:
                conn.execute("UPDATE documents SET state='indexed', error=?, indexed_at=CURRENT_TIMESTAMP WHERE id=?", (semantic_error, doc_id))
                conn.execute("UPDATE index_jobs SET processed_files=processed_files+1, indexed_files=indexed_files+1, current_path=? WHERE id=?", (str(path), job_id))
        except Exception as exc:
            with self.db.transaction() as conn:
                if existing:
                    conn.execute("UPDATE documents SET state='failed', error=? WHERE id=?", (str(exc), existing["id"]))
                else:
                    conn.execute("INSERT INTO documents(path,title,size,mtime_ns,state,error) VALUES (?,?,?,?,'failed',?)", (str(path), path.stem, stat.st_size, stat.st_mtime_ns, str(exc)))
                conn.execute("UPDATE index_jobs SET processed_files=processed_files+1, failed_files=failed_files+1, current_path=? WHERE id=?", (str(path), job_id))

    def _backfill_document(self, document_id: int, batch_size: int = 128) -> None:
        """Embed passages created while semantic models were unavailable."""
        while True:
            with self.db.connect() as conn:
                rows = conn.execute(
                    "SELECT p.id, p.text FROM passages p JOIN pages pg ON pg.id=p.page_id "
                    "WHERE pg.document_id=? AND p.active=1 AND p.embedding_id IS NULL "
                    "ORDER BY p.id LIMIT ?",
                    (document_id, batch_size),
                ).fetchall()
            if not rows:
                break
            ids = [int(row["id"]) for row in rows]
            embeddings = self.models.encode([row["text"] for row in rows])
            self.vectors.add(ids, embeddings)
            with self.db.transaction() as conn:
                conn.executemany(
                    "UPDATE passages SET embedding_id=? WHERE id=?",
                    [(passage_id, passage_id) for passage_id in ids],
                )
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE documents SET error=NULL, indexed_at=CURRENT_TIMESTAMP WHERE id=?",
                (document_id,),
            )

    def backfill_missing(self) -> int:
        """Backfill every active document and return the number of repaired documents."""
        if not self._running.acquire(blocking=False):
            return 0
        repaired = 0
        try:
            with self.db.connect() as conn:
                documents = conn.execute(
                    "SELECT DISTINCT pg.document_id FROM passages p "
                    "JOIN pages pg ON pg.id=p.page_id JOIN documents d ON d.id=pg.document_id "
                    "WHERE p.active=1 AND p.embedding_id IS NULL AND d.active=1"
                ).fetchall()
            for row in documents:
                self._backfill_document(int(row["document_id"]))
                repaired += 1
            return repaired
        finally:
            self._running.release()
