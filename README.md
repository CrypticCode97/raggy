# Raggy Evidence Search

Raggy is a local-first search engine for PDF evidence. It returns ranked original pages with exact provenance, highlights, and direct navigation instead of generating answers.

## Quick start

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and a modern browser.

```bash
git clone https://github.com/CrypticCode97/raggy.git
cd raggy
uv sync
uv run raggy serve
```

Open `http://127.0.0.1:7734`. The first-run wizard detects your CPU/GPU, offers
quality, balanced, and exact-only search profiles, and installs the selected OCR
languages. Add one or more PDF folders in **Library**, then choose **Build
incremental index**.

After setup, indexing and search work completely offline. Original PDFs stay in
place and are never modified.

Caution: Loads a huge amount of dependencies.

## Common commands

```bash
# Start without opening a browser
uv run raggy serve --no-open-browser

# Use a different port
uv run raggy serve --port 8080

# Show corpus statistics
uv run raggy benchmark
```

Press `Ctrl+C` to stop the service.

## Headless setup

The graphical wizard is recommended. For unattended installations:

```bash
# Balanced semantic search with English and German OCR
uv sync --extra ml
uv run raggy setup --profile fallback --languages eng+deu

# Maximum-quality BGE models on a capable GPU
uv run raggy setup --profile quality --languages eng+deu
```

## Development

```bash
uv sync --extra dev --extra ml
uv run pytest -q
uv run ruff check src tests
cd frontend && npm install && npm run build
```

## Architecture

- FastAPI, SQLite/FTS5, PyMuPDF and FAISS backend
- React, Tailwind CSS and PDF.js frontend
- Page-level provenance with passage-level semantic retrieval
- Native extraction with selective 300-DPI OCR fallback
- Manual incremental rescans with changed/deleted document detection

Application data is stored in the platform-specific user data directory.

## License

AGPL-3.0-or-later. PyMuPDF is also used under the AGPL; proprietary distribution requires appropriate commercial licensing from Artifex.
