from io import BytesIO
from pathlib import Path

import pytest

import app
import app1
from neural_script_decoding import web


@pytest.fixture()
def client(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    app1.app.config.update(
        TESTING=True,
        UPLOAD_FOLDER=str(upload_dir),
        WTF_CSRF_ENABLED=False,
    )

    monkeypatch.setattr(
        app1.ocr_engine,
        "process_with_all_engines",
        lambda _path: [
            {"engine": "tesseract", "text": "ok", "confidence": 91.2},
            {"engine": "easyocr", "text": "rich raw text output", "confidence": 85.0},
        ],
    )
    monkeypatch.setattr(
        app1.ocr_engine,
        "available_summary",
        lambda: {
            "tesseract": {"enabled": True, "installed": True, "ready": True, "reason": "Ready"},
            "easyocr": {"enabled": False, "installed": False, "ready": False, "reason": "Disabled by configuration"},
            "trocr": {"enabled": False, "installed": False, "ready": False, "reason": "Disabled by configuration"},
        },
    )
    monkeypatch.setattr(web, "correct_text_with_llm", lambda text, _config: f"fixed {text}")
    monkeypatch.setattr(web, "ollama_status", lambda _config: {"configured": True, "reachable": False, "reason": "Connection refused"})

    def fake_pdf(text, output_path, _font_path):
        Path(output_path).write_text(text, encoding="utf-8")

    monkeypatch.setattr(web, "generate_pdf", fake_pdf)

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

    upload_dir = Path(app1.app.config["UPLOAD_FOLDER"])
    files = sorted(path.name for path in upload_dir.iterdir())
    assert files == [name for name in files if name.endswith(".pdf")]
    assert len(files) == 1


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
    assert payload["system"]["engines"]["tesseract"]["ready"] is True
    assert payload["selected_engine"] == "easyocr"

    upload_dir = Path(app1.app.config["UPLOAD_FOLDER"])
    assert list(upload_dir.iterdir()) == []


def test_download_json_returns_attachment(client):
    response = client.post(
        "/download-json",
        json={"file": "sample.jpg", "raw_text": "a", "corrected_text": "b", "ocr_results": [], "system": {}},
    )

    assert response.status_code == 200
    assert response.mimetype == "application/json"
    assert "attachment; filename=\"sample.json\"" in response.headers["Content-Disposition"]


def test_api_ocr_requires_file(client):
    response = client.post("/api/ocr", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "No file provided"


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
