import json
import sqlite3
import time
from io import BytesIO
from pathlib import Path

import pytest
from flask import Flask

import app
import app1
from neural_script_decoding.background_jobs import OCRJobManager
from neural_script_decoding.config import apply_config
from neural_script_decoding import services
from neural_script_decoding import web
from neural_script_decoding.storage import init_db


@pytest.fixture()
def client(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    database_path = tmp_path / "runs.sqlite3"

    app1.app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(upload_dir),
        DATABASE_PATH=str(database_path),
        MAX_SAVED_RUNS=100,
        BACKGROUND_OCR_WORKERS=1,
        WTF_CSRF_ENABLED=False,
    )
    init_db(app1.app.config["DATABASE_PATH"])
    app1.app.extensions["ocr_job_manager"] = OCRJobManager(config=app1.app.config, ocr_engine=app1.ocr_engine)

    monkeypatch.setattr(
        app1.ocr_engine,
        "process_with_all_engines",
        lambda _path: [
            {"engine": "tesseract", "text": "ok", "confidence": 91.2},
            {"engine": "easyocr", "text": "rich raw text output", "confidence": 85.0},
        ],
    )
    def fake_overlay(_source, output):
        Path(output).write_bytes(b"overlay")
        return {
            "overlay_path": output,
            "regions": [
                {"text": "rich", "confidence": 92.0, "x": 10, "y": 10, "w": 40, "h": 20},
                {"text": "text", "confidence": 88.0, "x": 60, "y": 10, "w": 40, "h": 20},
            ],
        }

    monkeypatch.setattr(app1.ocr_engine, "create_tesseract_overlay", fake_overlay)
    monkeypatch.setattr(
        app1.ocr_engine,
        "available_summary",
        lambda: {
            "tesseract": {"enabled": True, "installed": True, "ready": True, "reason": "Ready"},
            "easyocr": {"enabled": False, "installed": False, "ready": False, "reason": "Disabled by configuration"},
            "trocr": {"enabled": False, "installed": False, "ready": False, "reason": "Disabled by configuration"},
        },
    )
    monkeypatch.setattr(
        services,
        "correct_text_with_llm",
        lambda text, _config: {
            "text": f"fixed {text}",
            "used_llm": True,
            "status": "corrected",
            "reason": "Corrected with Ollama",
        },
    )
    monkeypatch.setattr(web, "ollama_status", lambda _config: {"configured": True, "reachable": False, "reason": "Connection refused"})

    def fake_pdf(text, output_path, _font_path):
        Path(output_path).write_text(text, encoding="utf-8")

    monkeypatch.setattr(services, "generate_pdf", fake_pdf)
    monkeypatch.setattr(services, "persist_preview", lambda source, dest: Path(dest).write_bytes(Path(source).read_bytes()))

    with app1.app.test_client() as test_client:
        yield test_client


def test_index_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Neural Script Decoding" in response.data


def test_app_entrypoint_exports_same_flask_app():
    assert app.app is app1.app


def test_upload_generates_pdf_and_cleans_temp_files(client):
    response = client.post(
        "/upload",
        data={"file": (BytesIO(b"fake-image-bytes"), "note.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"OCR result board" in response.data
    assert b"Download JSON" in response.data
    assert b"Open Saved Run" in response.data
    assert b"/preview/" in response.data
    assert b"Detected words and confidence" in response.data

    upload_dir = Path(app1.app.config["UPLOAD_FOLDER"])
    files = sorted(path.name for path in upload_dir.iterdir())
    assert len([name for name in files if name.endswith(".pdf")]) == 1
    assert len([name for name in files if "_preview." in name]) == 1


def test_api_ocr_returns_json_and_cleans_temp_files(client):
    response = client.post(
        "/api/ocr",
        data={"file": (BytesIO(b"fake-image-bytes"), "note.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["raw_text"] == "rich raw text output"
    assert payload["corrected_text"] == "fixed rich raw text output"
    assert payload["correction"]["used_llm"] is True
    assert payload["system"]["engines"]["tesseract"]["ready"] is True
    assert payload["selected_engine"] == "easyocr"
    assert "/download/" in payload["pdf_url"]
    assert "/preview/" in payload["preview_url"]
    assert payload["regions"][0]["text"] == "rich"

    upload_dir = Path(app1.app.config["UPLOAD_FOLDER"])
    remaining = sorted(path.name for path in upload_dir.iterdir())
    assert any(name.endswith(".pdf") for name in remaining)
    assert any("_preview." in name for name in remaining)


def test_api_ocr_saved_run_has_downloadable_pdf(client):
    response = client.post(
        "/api/ocr",
        data={"file": (BytesIO(b"fake-image-bytes"), "note.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()

    download = client.get(payload["pdf_url"])
    assert download.status_code == 200
    assert download.mimetype == "application/pdf"


def test_api_ocr_async_completes_in_background(client):
    queued = client.post(
        "/api/ocr/jobs",
        data={"file": (BytesIO(b"fake-image-bytes"), "note.jpg")},
        content_type="multipart/form-data",
    )

    assert queued.status_code == 202
    payload = queued.get_json()
    assert payload["success"] is True
    assert payload["status"] in {"queued", "running"}
    assert payload["job_id"]

    completed = None
    for _ in range(40):
        status = client.get(payload["status_url"])
        assert status.status_code == 200
        completed = status.get_json()
        if completed["status"] == "completed":
            break
        time.sleep(0.05)

    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["corrected_text"] == "fixed rich raw text output"
    assert completed["run_id"] == 1
    assert "/download/" in completed["pdf_url"]

    download = client.get(completed["pdf_url"])
    assert download.status_code == 200


def test_api_ocr_async_missing_job_returns_404(client):
    response = client.get("/api/ocr/jobs/missing-job")

    assert response.status_code == 404
    assert response.get_json()["error"] == "Job not found"


def test_download_json_returns_attachment(client):
    response = client.post(
        "/download-json",
        json={"file": "sample.jpg", "raw_text": "a", "corrected_text": "b", "ocr_results": [], "system": {}},
    )

    assert response.status_code == 200
    assert response.mimetype == "application/json"
    assert "attachment; filename=\"sample.json\"" in response.headers["Content-Disposition"]


def test_download_json_accepts_form_payload_without_javascript(client):
    response = client.post(
        "/download-json",
        data={"payload": json.dumps({"file": "sample.jpg", "raw_text": "a", "corrected_text": "b", "ocr_results": [], "system": {}})},
    )

    assert response.status_code == 200
    assert response.mimetype == "application/json"
    assert "attachment; filename=\"sample.json\"" in response.headers["Content-Disposition"]


def test_history_pages_render_saved_runs(client):
    response = client.post(
        "/upload",
        data={"file": (BytesIO(b"fake-image-bytes"), "note.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200

    history = client.get("/history")
    assert history.status_code == 200
    assert b"OCR history" in history.data
    assert b"note.jpg" in history.data
    assert b"/preview/" in history.data

    detail = client.get("/history/1")
    assert detail.status_code == 200
    assert b"Stored Run" in detail.data
    assert b"rich raw text output" in detail.data
    assert b"Corrected with Ollama" in detail.data
    assert b"/preview/" in detail.data
    assert b"Detected words and confidence" in detail.data


def test_upload_falls_back_to_raw_text_when_llm_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(
        services,
        "correct_text_with_llm",
        lambda text, _config: {
            "text": text,
            "used_llm": False,
            "status": "fallback",
            "reason": "Ollama unavailable (Connection refused); using raw OCR text",
        },
    )

    response = client.post(
        "/upload",
        data={"file": (BytesIO(b"fake-image-bytes"), "note.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"rich raw text output" in response.data
    assert b"Ollama unavailable" in response.data
    assert b"[Error: LLM offline" not in response.data

    detail = client.get("/history/1")
    assert detail.status_code == 200
    assert b"Ollama unavailable" in detail.data
    assert b"[Error: LLM offline" not in detail.data


def test_diagnostics_page_renders_runtime_state(client):
    response = client.get("/diagnostics")

    assert response.status_code == 200
    assert b"Environment and dependency status" in response.data
    assert b"phi3:mini" in response.data
    assert b"uploads" in response.data


def test_health_reports_background_job_summary(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["background_jobs"]["total"] == 0
    assert payload["background_jobs"]["queued"] == 0


def test_history_filters_and_delete(client):
    client.post(
        "/upload",
        data={"file": (BytesIO(b"fake-image-bytes"), "note.jpg")},
        content_type="multipart/form-data",
    )
    client.post(
        "/upload",
        data={"file": (BytesIO(b"fake-image-bytes"), "other.jpg")},
        content_type="multipart/form-data",
    )

    filtered = client.get("/history?q=other&engine=easyocr")
    assert filtered.status_code == 200
    assert b"other.jpg" in filtered.data
    assert b"note.jpg" not in filtered.data

    deleted = client.post("/history/1/delete", follow_redirects=True)
    assert deleted.status_code == 200
    assert b"Saved run deleted" in deleted.data
    assert client.get("/history/1").status_code == 404


def test_history_prunes_old_runs(client):
    app1.app.config["MAX_SAVED_RUNS"] = 1

    client.post(
        "/upload",
        data={"file": (BytesIO(b"fake-image-bytes"), "first.jpg")},
        content_type="multipart/form-data",
    )
    client.post(
        "/upload",
        data={"file": (BytesIO(b"fake-image-bytes"), "second.jpg")},
        content_type="multipart/form-data",
    )

    history = client.get("/history")
    assert history.status_code == 200
    assert b"second.jpg" in history.data
    assert b"first.jpg" not in history.data


def test_history_detail_normalizes_legacy_payload(client):
    payload = {
        "file": "legacy.jpg",
        "original_file": "legacy.jpg",
        "selected_engine": "tesseract",
        "raw_text": "legacy raw",
        "corrected_text": "legacy corrected",
        "ocr_results": [],
        "system": {},
    }
    with sqlite3.connect(app1.app.config["DATABASE_PATH"]) as connection:
        connection.execute(
            """
            INSERT INTO ocr_runs (
                file_name, original_file_name, selected_engine, raw_text, corrected_text,
                pdf_file_name, preview_file_name, overlay_file_name, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy.jpg",
                "legacy.jpg",
                "tesseract",
                "legacy raw",
                "legacy corrected",
                None,
                None,
                None,
                json.dumps(payload),
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()

    response = client.get("/history/1")

    assert response.status_code == 200
    assert b"legacy corrected" in response.data
    assert b"Stored Run" in response.data


def test_api_ocr_requires_file(client):
    response = client.post("/api/ocr", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "No file provided"


def test_apply_config_requires_secret_key_in_production(monkeypatch):
    flask_app = Flask(__name__)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY must be set"):
        apply_config(flask_app)


def test_apply_config_accepts_secret_key_in_production(monkeypatch):
    flask_app = Flask(__name__)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret")

    apply_config(flask_app)

    assert flask_app.secret_key == "test-secret"


def test_health_reports_engine_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert set(data["system"]["engines"]) == {"tesseract", "easyocr", "trocr"}
    assert data["system"]["ollama"]["reachable"] is False


def test_index_shows_missing_engine_status(client, monkeypatch):
    monkeypatch.setattr(
        app1.ocr_engine,
        "available_summary",
        lambda: {
            "tesseract": {"enabled": False, "installed": True, "ready": False, "reason": "Disabled by configuration"},
            "easyocr": {"enabled": False, "installed": False, "ready": False, "reason": "Disabled by configuration"},
            "trocr": {"enabled": False, "installed": False, "ready": False, "reason": "Disabled by configuration"},
        },
    )
    monkeypatch.setattr(web, "ollama_status", lambda _config: {"configured": True, "reachable": False, "reason": "Offline"})

    response = client.get("/")

    assert response.status_code == 200
    assert response.data.count(b"Unavailable") == 3
    assert b"Offline" in response.data


def test_download_blocks_path_traversal(client):
    response = client.get("/download/../../app1.py")

    assert response.status_code == 404


def test_invalid_file_type_rejected(client):
    response = client.post(
        "/upload",
        data={"file": (BytesIO(b"not-an-image"), "note.txt")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Allowed file types are png, jpg, jpeg, gif, bmp" in response.data
