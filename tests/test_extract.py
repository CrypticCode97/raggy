from pathlib import Path

import pymupdf

from raggy.extract import chunk_words, extract_pdf, normalize_text, unpack_layout


def test_normalize_text():
    assert normalize_text("  ISO\u00a0４７６２  ") == "iso 4762"


def test_extract_keeps_page_and_coordinates(tmp_path: Path):
    path = tmp_path / "catalogue.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Marine fastener ISO 4762 A4-80 corrosion resistance")
    document.save(path)
    document.close()

    pages = extract_pdf(path)
    assert len(pages) == 1
    assert "ISO 4762" in pages[0].text
    words = unpack_layout(pages[0].layout)
    assert all(0 <= word["x"] <= 1 and 0 <= word["y"] <= 1 for word in words)
    assert chunk_words(pages[0].layout)[0][0].startswith("Marine fastener")
