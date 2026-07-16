import json
import zlib
from pathlib import Path

import numpy as np

from raggy.config import Settings
from raggy.db import Database
from raggy.indexer import IndexCoordinator


class FakeModels:
    def encode(self, texts):
        vectors = np.ones((len(texts), 4), dtype="float32")
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class FakeVectors:
    def __init__(self):
        self.ids = []

    def add(self, ids, vectors):
        self.ids.extend(ids)


def test_unchanged_document_backfills_missing_embeddings(tmp_path: Path):
    source = tmp_path / "evidence.pdf"
    source.write_bytes(b"pdf fixture")
    stat = source.stat()
    settings = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    db = Database(settings.db_path)
    db.initialize()
    layout = zlib.compress(json.dumps([]).encode())
    with db.transaction() as conn:
        document_id = conn.execute(
            "INSERT INTO documents(path,title,size,mtime_ns,page_count,state,error) "
            "VALUES (?,?,?,?,1,'indexed','models unavailable')",
            (str(source), "Evidence", stat.st_size, stat.st_mtime_ns),
        ).lastrowid
        page_id = conn.execute(
            "INSERT INTO pages(document_id,page_number,text,normalized_text,width,height,"
            "extraction_method,layout_json) VALUES (?,1,'marine steel','marine steel',1,1,'native',?)",
            (document_id, layout),
        ).lastrowid
        passage_id = conn.execute(
            "INSERT INTO passages(page_id,ordinal,text,word_start,word_end) VALUES (?,0,'marine steel',0,2)",
            (page_id,),
        ).lastrowid
        job_id = conn.execute("INSERT INTO index_jobs(state) VALUES ('running')").lastrowid
    vectors = FakeVectors()
    coordinator = IndexCoordinator(db, settings, FakeModels(), vectors)
    coordinator._process_one(source, job_id)
    with db.connect() as conn:
        passage = conn.execute("SELECT embedding_id FROM passages WHERE id=?", (passage_id,)).fetchone()
        document = conn.execute("SELECT error FROM documents WHERE id=?", (document_id,)).fetchone()
    assert passage["embedding_id"] == passage_id
    assert document["error"] is None
    assert vectors.ids == [passage_id]
