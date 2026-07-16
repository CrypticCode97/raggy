from __future__ import annotations

import json
import urllib.request
import webbrowser

import typer
import uvicorn
from huggingface_hub import snapshot_download

from .config import Settings

app = typer.Typer(help="Local-first PDF evidence search")


@app.command()
def setup(
    profile: str = typer.Option("quality", help="quality or fallback"),
    languages: str = typer.Option("eng", help="Tesseract codes separated by +"),
):
    """Download model snapshots and selected OCR language data for offline use."""
    settings = Settings(model_profile=profile, ocr_languages=languages)
    settings.ensure_dirs()
    model_names = (
        [settings.embedding_model, settings.reranker_model]
        if profile == "quality"
        else [settings.fallback_embedding_model, settings.fallback_reranker_model]
    )
    for model in model_names:
        typer.echo(f"Downloading {model}...")
        snapshot_download(model)
    tessdata = settings.data_dir / "tessdata"
    tessdata.mkdir(exist_ok=True)
    for language in languages.split("+"):
        destination = tessdata / f"{language}.traineddata"
        typer.echo(f"Downloading OCR language {language}...")
        urllib.request.urlretrieve(
            f"https://github.com/tesseract-ocr/tessdata_fast/raw/main/{language}.traineddata",
            destination,
        )
    config = {"model_profile": profile, "ocr_languages": languages, "tessdata": str(tessdata)}
    (settings.data_dir / "setup.json").write_text(json.dumps(config, indent=2))
    typer.echo(f"Offline assets are ready in {settings.data_dir}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 7734, open_browser: bool = True):
    """Start the local search service."""
    if open_browser:
        webbrowser.open(f"http://{host}:{port}")
    uvicorn.run("raggy.api:app", host=host, port=port, reload=False)


@app.command()
def benchmark(iterations: int = 20):
    """Print local index statistics; query latency benchmarking is exposed by the API."""
    settings = Settings()
    from .db import Database
    db = Database(settings.db_path)
    db.initialize()
    with db.connect() as conn:
        result = dict(conn.execute("SELECT COUNT(*) documents, COALESCE(SUM(page_count),0) pages FROM documents WHERE active=1").fetchone())
    result["iterations"] = iterations
    typer.echo(json.dumps(result, indent=2))
