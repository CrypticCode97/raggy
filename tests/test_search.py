import json
import zlib
from pathlib import Path

from raggy.db import Database
from raggy.search import SearchService


class NoModels:
    def encode(self, *args, **kwargs):
        raise RuntimeError("models offline")

    def rerank(self, *args, **kwargs):
        raise RuntimeError("models offline")


class NoVectors:
    def search(self, *args, **kwargs):
        return []


def layout(text: str) -> bytes:
    words = []
    position = 0
    for index, word in enumerate(text.split()):
        words.append({"i": index, "t": word, "s": position, "e": position + len(word), "x": index / 20, "y": 0.1, "w": 0.04, "h": 0.02, "b": 0, "l": 0, "o": index})
        position += len(word) + 1
    return zlib.compress(json.dumps(words).encode())


def seeded(tmp_path: Path) -> SearchService:
    db = Database(tmp_path / "test.sqlite")
    db.initialize()
    with db.transaction() as conn:
        doc = conn.execute("INSERT INTO documents(path,title,size,mtime_ns,page_count,state) VALUES ('/tmp/a.pdf','Fastener Catalogue',1,1,2,'indexed')").lastrowid
        one = "Marine fastener ISO 4762 A4-80 corrosion resistance"
        two = "Unrelated carbon steel bolt"
        conn.execute("INSERT INTO pages(document_id,page_number,text,normalized_text,width,height,extraction_method,layout_json) VALUES (?,?,?,?,100,100,'native',?)", (doc, 182, one, one.casefold(), layout(one)))
        conn.execute("INSERT INTO pages(document_id,page_number,text,normalized_text,width,height,extraction_method,layout_json) VALUES (?,?,?,?,100,100,'native',?)", (doc, 183, two, two.casefold(), layout(two)))
    return SearchService(db, NoModels(), NoVectors())


def test_exact_is_exhaustive_and_provenanced(tmp_path: Path):
    output = seeded(tmp_path).search("ISO 4762", "exact")
    assert output.total == 1
    assert output.results[0]["page_number"] == 182
    assert output.results[0]["title"] == "Fastener Catalogue"
    assert output.results[0]["highlights"]


def test_hybrid_falls_back_to_lexical(tmp_path: Path):
    output = seeded(tmp_path).search("corrosion resistance", "hybrid")
    assert output.results[0]["page_number"] == 182
    assert output.warning == "models offline"

