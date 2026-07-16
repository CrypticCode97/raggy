from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient

from raggy.api import create_app
from raggy.config import Settings


def test_health_and_source_validation(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json() == {"status": "ok", "offline": True}
        response = client.post("/api/sources", json={"path": str(tmp_path)})
        assert response.status_code == 201
        assert client.get("/api/sources").json()[0]["path"] == str(tmp_path.resolve())
        assert client.post("/api/search", json={"query": "ISO 4762", "mode": "exact"}).status_code == 200


def test_setup_capabilities_and_exact_profile(tmp_path: Path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", cache_dir=tmp_path / "cache")
    monkeypatch.setattr("raggy.setup_service.urllib.request.urlretrieve", lambda url, destination: Path(destination).write_bytes(b"ocr"))
    monkeypatch.setattr(
        "raggy.setup_service.SetupCoordinator._verify_ocr",
        staticmethod(lambda tessdata, languages: None),
    )
    with TestClient(create_app(settings)) as client:
        status = client.get("/api/setup").json()
        assert {profile["id"] for profile in status["capabilities"]["profiles"]} == {"quality", "fallback", "exact"}
        assert "semantic_runtime_error" in status["capabilities"]
        assert "ocr_ready" in status["capabilities"]
        assert all(language["download_mb"] > 0 for language in status["capabilities"]["ocr_languages"])
        response = client.post("/api/setup", json={"profile": "exact", "languages": ["eng"], "install_runtime": False})
        assert response.status_code == 202
        for _ in range(50):
            job = client.get("/api/setup").json()["job"]
            if job["state"] != "running":
                break
            sleep(0.01)
        assert job["state"] == "complete", job
        assert (settings.data_dir / "setup.json").exists()
        assert client.get("/api/setup").json()["capabilities"]["ocr_ready"] is True
