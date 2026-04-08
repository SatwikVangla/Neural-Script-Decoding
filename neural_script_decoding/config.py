import os
from pathlib import Path


def apply_config(app):
    project_root = Path(__file__).resolve().parent.parent

    def resolve_path(value, default_relative):
        raw = Path(os.environ.get(value, str(project_root / default_relative))).expanduser()
        if raw.is_absolute():
            return str(raw)
        return str((project_root / raw).resolve())

    app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))
    app.config["UPLOAD_FOLDER"] = resolve_path("UPLOAD_FOLDER", Path("static") / "uploads")
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "16")) * 1024 * 1024
    app.config["OLLAMA_URL"] = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    app.config["OLLAMA_MODEL"] = os.environ.get("OLLAMA_MODEL", "mistral")
    app.config["PDF_FONT_PATH"] = resolve_path("PDF_FONT_PATH", Path("static") / "DejaVuSans.ttf")
    app.config["DATABASE_PATH"] = resolve_path("DATABASE_PATH", Path("instance") / "neural_script_decoding.sqlite3")
    app.config["APP_HOST"] = os.environ.get("APP_HOST", "0.0.0.0")
    app.config["APP_PORT"] = int(os.environ.get("APP_PORT", "5000"))
    app.config["APP_DEBUG"] = os.environ.get("APP_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    app.config["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "INFO").upper()
    app.config["MAX_SAVED_RUNS"] = int(os.environ.get("MAX_SAVED_RUNS", "100"))
    app.config["ENABLE_TESSERACT"] = os.environ.get("ENABLE_TESSERACT", "true").lower() in {"1", "true", "yes", "on"}
    app.config["ENABLE_EASYOCR"] = os.environ.get("ENABLE_EASYOCR", "false").lower() in {"1", "true", "yes", "on"}
    app.config["ENABLE_TROCR"] = os.environ.get("ENABLE_TROCR", "false").lower() in {"1", "true", "yes", "on"}
    app.config["OLLAMA_STATUS_TIMEOUT"] = float(os.environ.get("OLLAMA_STATUS_TIMEOUT", "1.0"))
