from io import BytesIO
from pathlib import Path

import pytest

import app1


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
            {"engine": "tesseract", "text": "raw text", "confidence": 91.2},
            {"engine": "easyocr", "text": "raw text", "confidence": 85.0},
        ],
    )
    monkeypatch.setattr(app1.ocr_engine, "combine_results", lambda results: results[0]["text"])
    monkeypatch.setattr(app1, "correct_text_with_llm", lambda text: f"fixed {text}")

    def fake_pdf(text, output_path):
        Path(output_path).write_text(text, encoding="utf-8")

    monkeypatch.setattr(app1, "generate_pdf", fake_pdf)

    with app1.app.test_client() as test_client:
        yield test_client


def test_index_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Neural Script Decoding" in response.data


def test_upload_generates_pdf_and_cleans_temp_files(client):
    response = client.post(
        "/upload",
        data={"file": (BytesIO(b"fake-image-bytes"), "note.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"OCR result board" in response.data

    upload_dir = Path(app1.app.config["UPLOAD_FOLDER"])
    files = sorted(path.name for path in upload_dir.iterdir())
    assert files == [name for name in files if name.endswith(".pdf")]
    assert len(files) == 1


def test_health_reports_engine_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert set(data["engines"]) == {"tesseract", "easyocr", "trocr"}


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
