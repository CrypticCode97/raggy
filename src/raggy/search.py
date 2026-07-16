from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from .db import Database
from .extract import normalize_text, unpack_layout
from .models import ModelManager
from .vectors import VectorIndex


def _fts_query(query: str) -> str:
    tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
    return " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-30, min(30, value))))


@dataclass(slots=True)
class SearchOutput:
    results: list[dict]
    total: int | None
    took_ms: float
    warning: str | None = None


class SearchService:
    def __init__(self, db: Database, models: ModelManager, vectors: VectorIndex):
        self.db = db
        self.models = models
        self.vectors = vectors

    def search(self, query: str, mode: str, page: int = 1, page_size: int = 10) -> SearchOutput:
        started = time.perf_counter()
        if mode == "exact":
            results, total = self._exact(query, page, page_size)
            return SearchOutput(results, total, (time.perf_counter() - started) * 1000)
        warning = None
        lexical = self._lexical(query, 200) if mode == "hybrid" else []
        semantic: list[tuple[int, float, str, int, int]] = []
        try:
            semantic = self._semantic(query, 200)
        except RuntimeError as exc:
            if mode == "semantic":
                raise
            warning = str(exc)
        fused: dict[int, dict] = {}
        for rank, (page_id, score) in enumerate(lexical, 1):
            fused.setdefault(page_id, {"rrf": 0.0, "passage": None, "word_start": 0, "word_end": 0})
            fused[page_id]["rrf"] += 1 / (60 + rank)
        for rank, (page_id, score, passage, ws, we) in enumerate(semantic, 1):
            item = fused.setdefault(page_id, {"rrf": 0.0, "passage": passage, "word_start": ws, "word_end": we})
            item["rrf"] += 1 / (60 + rank)
            if item["passage"] is None:
                item.update(passage=passage, word_start=ws, word_end=we)
        ranked = sorted(fused.items(), key=lambda item: item[1]["rrf"], reverse=True)[:40]
        if ranked:
            texts = [item[1]["passage"] or self._page_text(item[0]) for item in ranked]
            try:
                rerank_scores = self.models.rerank(query, texts)
            except RuntimeError as exc:
                rerank_scores = [0.0] * len(ranked)
                warning = str(exc)
            max_rrf = max(data["rrf"] for _, data in ranked) or 1
            scored = [(pid, 0.75 * _sigmoid(rr) + 0.25 * data["rrf"] / max_rrf, data) for (pid, data), rr in zip(ranked, rerank_scores, strict=True)]
            scored.sort(key=lambda item: item[1], reverse=True)
        else:
            scored = []
        selected = scored[(page - 1) * page_size:page * page_size]
        results = [self._result(pid, score, query, data) for pid, score, data in selected]
        return SearchOutput(results, None, (time.perf_counter() - started) * 1000, warning)

    def _exact(self, query: str, page: int, page_size: int) -> tuple[list[dict], int]:
        needle = normalize_text(query)
        if not needle:
            return [], 0
        with self.db.connect() as conn:
            candidates = conn.execute("SELECT p.id FROM pages_exact_fts f JOIN pages p ON p.id=f.rowid JOIN documents d ON d.id=p.document_id WHERE pages_exact_fts MATCH ? AND d.active=1 AND instr(p.normalized_text, ?) > 0 ORDER BY d.title, p.page_number", (f'"{needle.replace(chr(34), chr(34) * 2)}"', needle)).fetchall()
        total = len(candidates)
        ids = [row["id"] for row in candidates[(page - 1) * page_size:page * page_size]]
        return [self._result(pid, 1.0, query, {}) for pid in ids], total

    def _lexical(self, query: str, limit: int) -> list[tuple[int, float]]:
        fts = _fts_query(query)
        if not fts:
            return []
        with self.db.connect() as conn:
            rows = conn.execute("SELECT p.id, bm25(pages_fts) score FROM pages_fts JOIN pages p ON p.id=pages_fts.rowid JOIN documents d ON d.id=p.document_id WHERE pages_fts MATCH ? AND d.active=1 ORDER BY score LIMIT ?", (fts, limit)).fetchall()
        return [(int(row["id"]), -float(row["score"])) for row in rows]

    def _semantic(self, query: str, limit: int) -> list[tuple[int, float, str, int, int]]:
        vector = self.models.encode([query], query=True)
        hits = self.vectors.search(vector, limit * 2)
        if not hits:
            with self.db.connect() as conn:
                counts = conn.execute(
                    "SELECT COUNT(*) total, COUNT(embedding_id) embedded FROM passages WHERE active=1"
                ).fetchone()
            if counts["total"] and counts["embedded"] < counts["total"]:
                raise RuntimeError(
                    f"Semantic index is being built ({counts['embedded']} of {counts['total']} passages ready)."
                )
            return []
        best: dict[int, tuple[int, float, str, int, int]] = {}
        with self.db.connect() as conn:
            for passage_id, score in hits:
                row = conn.execute("SELECT p.page_id,p.text,p.word_start,p.word_end FROM passages p JOIN pages pg ON pg.id=p.page_id JOIN documents d ON d.id=pg.document_id WHERE p.id=? AND p.active=1 AND d.active=1", (passage_id,)).fetchone()
                if row and (row["page_id"] not in best or score > best[row["page_id"]][1]):
                    best[row["page_id"]] = (row["page_id"], score, row["text"], row["word_start"], row["word_end"])
        return sorted(best.values(), key=lambda item: item[1], reverse=True)[:limit]

    def _page_text(self, page_id: int) -> str:
        with self.db.connect() as conn:
            row = conn.execute("SELECT text FROM pages WHERE id=?", (page_id,)).fetchone()
        return row["text"] if row else ""

    def _result(self, page_id: int, score: float, query: str, data: dict) -> dict:
        with self.db.connect() as conn:
            row = conn.execute("SELECT p.*,d.title,d.path,d.error document_warning FROM pages p JOIN documents d ON d.id=p.document_id WHERE p.id=?", (page_id,)).fetchone()
        words = unpack_layout(row["layout_json"])
        start, end = data.get("word_start", 0), data.get("word_end", min(len(words), 60))
        normalized_terms = set(normalize_text(query).split())
        boxes = [{"x": w["x"], "y": w["y"], "w": w["w"], "h": w["h"]} for w in words if normalize_text(w["t"]) in normalized_terms]
        if not boxes and end > start:
            boxes = [{"x": w["x"], "y": w["y"], "w": w["w"], "h": w["h"]} for w in words[start:end]]
        snippet_words = words[max(0, start - 12):min(len(words), max(end, start + 50) + 12)]
        return {"page_id": page_id, "document_id": row["document_id"], "title": row["title"], "path": row["path"], "page_number": row["page_number"], "score": round(score, 6), "snippet": " ".join(w["t"] for w in snippet_words), "highlights": boxes, "extraction_method": row["extraction_method"], "warning": row["warning"] or row["document_warning"], "pdf_url": f"/api/documents/{row['document_id']}/pdf#page={row['page_number']}"}
