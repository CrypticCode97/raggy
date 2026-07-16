from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class ExtractedPage:
    page_number: int
    text: str
    normalized_text: str
    width: float
    height: float
    method: str
    layout: bytes
    warning: str | None = None


def _pack_words(raw_words: list[tuple], width: float, height: float) -> tuple[str, bytes]:
    words: list[dict] = []
    text_parts: list[str] = []
    char_at = 0
    for index, word in enumerate(raw_words):
        x0, y0, x1, y1, value, block, line, order = word[:8]
        if not value.strip():
            continue
        if text_parts:
            text_parts.append(" ")
            char_at += 1
        start = char_at
        text_parts.append(value)
        char_at += len(value)
        words.append({
            "i": index, "t": value, "s": start, "e": char_at,
            "x": round(x0 / width, 6), "y": round(y0 / height, 6),
            "w": round((x1 - x0) / width, 6), "h": round((y1 - y0) / height, 6),
            "b": block, "l": line, "o": order,
        })
    text = "".join(text_parts)
    return text, zlib.compress(json.dumps(words, ensure_ascii=False).encode(), level=6)


def unpack_layout(blob: bytes) -> list[dict]:
    return json.loads(zlib.decompress(blob))


def extract_pdf(path: Path, languages: str = "eng", tessdata: Path | None = None) -> list[ExtractedPage]:
    result: list[ExtractedPage] = []
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("PDF is password protected")
        for page_index, page in enumerate(document):
            rect = page.rect
            words = page.get_text("words", sort=True)
            native_text = " ".join(str(w[4]) for w in words)
            image_area = 0.0
            for image in page.get_images(full=True):
                try:
                    for image_rect in page.get_image_rects(image[0]):
                        image_area += image_rect.get_area()
                except Exception:
                    continue
            coverage = image_area / max(rect.get_area(), 1)
            needs_ocr = len(normalize_text(native_text)) < 50 or (len(words) < 10 and coverage >= 0.5)
            method = "native"
            warning = None
            if needs_ocr:
                try:
                    textpage = page.get_textpage_ocr(
                        language=languages,
                        dpi=300,
                        full=len(normalize_text(native_text)) < 10,
                        tessdata=str(tessdata) if tessdata else None,
                    )
                    words = page.get_text("words", textpage=textpage, sort=True)
                    method = "ocr"
                except Exception as exc:
                    warning = f"OCR unavailable: {exc}"
            text, packed = _pack_words(words, rect.width, rect.height)
            result.append(ExtractedPage(
                page_number=page_index + 1,
                text=text,
                normalized_text=normalize_text(text),
                width=rect.width,
                height=rect.height,
                method=method,
                layout=packed,
                warning=warning,
            ))
    return result


def chunk_words(layout_blob: bytes, max_words: int = 260, overlap: int = 40) -> list[tuple[str, int, int]]:
    words = unpack_layout(layout_blob)
    if not words:
        return []
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append((" ".join(w["t"] for w in words[start:end]), start, end))
        if end == len(words):
            break
        start = max(start + 1, end - overlap)
    return chunks
