import os
from pathlib import Path


TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name, default):
    return os.environ.get(name, default).lower() in TRUTHY


def apply_config(app):
    project_root = Path(__file__).resolve().parent.parent

    def resolve_path(value, default_relative):
        raw = Path(os.environ.get(value, str(project_root / default_relative))).expanduser()
        if raw.is_absolute():
            return str(raw)
        return str((project_root / raw).resolve())

    app.config["APP_ENV"] = os.environ.get("APP_ENV", "development").lower()
    app.config["APP_DEBUG"] = _env_flag("APP_DEBUG", "false")
    secret_key = os.environ.get("FLASK_SECRET_KEY")
    if app.config["APP_ENV"] == "production" and not secret_key:
        raise RuntimeError("FLASK_SECRET_KEY must be set when APP_ENV=production")

    app.secret_key = secret_key or os.urandom(32)
    app.config["UPLOAD_FOLDER"] = resolve_path("UPLOAD_FOLDER", Path("static") / "uploads")
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "16")) * 1024 * 1024
    app.config["OLLAMA_URL"] = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    app.config["OLLAMA_MODEL"] = os.environ.get("OLLAMA_MODEL", "phi3:mini")
    app.config["PDF_FONT_PATH"] = resolve_path("PDF_FONT_PATH", Path("static") / "DejaVuSans.ttf")
    app.config["DATABASE_PATH"] = resolve_path("DATABASE_PATH", Path("instance") / "neural_script_decoding.sqlite3")
    app.config["APP_HOST"] = os.environ.get("APP_HOST", "0.0.0.0")
    app.config["APP_PORT"] = int(os.environ.get("APP_PORT", "5000"))
    app.config["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "INFO").upper()
    app.config["MAX_SAVED_RUNS"] = int(os.environ.get("MAX_SAVED_RUNS", "100"))
    app.config["MAX_STORED_JOBS"] = int(os.environ.get("MAX_STORED_JOBS", "200"))
    app.config["JOB_RETENTION_DAYS"] = int(os.environ.get("JOB_RETENTION_DAYS", "7"))
    app.config["BACKGROUND_OCR_WORKERS"] = int(os.environ.get("BACKGROUND_OCR_WORKERS", "2"))
    app.config["BACKGROUND_JOB_TIMEOUT"] = int(os.environ.get("BACKGROUND_JOB_TIMEOUT", "300"))
    app.config["OCR_QUEUE_BACKEND"] = os.environ.get("OCR_QUEUE_BACKEND", "local").lower()
    app.config["OCR_QUEUE_NAME"] = os.environ.get("OCR_QUEUE_NAME", "ocr")
    app.config["REDIS_URL"] = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    app.config["AUTH_REQUIRED"] = _env_flag("AUTH_REQUIRED", "false")
    app.config["ADMIN_USERNAME"] = os.environ.get("ADMIN_USERNAME", "admin")
    app.config["ADMIN_PASSWORD_HASH"] = os.environ.get("ADMIN_PASSWORD_HASH", "")
    app.config["API_KEY"] = os.environ.get("API_KEY", "")
    app.config["API_RATE_LIMIT"] = int(os.environ.get("API_RATE_LIMIT", "30"))
    app.config["API_RATE_WINDOW_SECONDS"] = int(os.environ.get("API_RATE_WINDOW_SECONDS", "60"))
    app.config["ENABLE_TESSERACT"] = _env_flag("ENABLE_TESSERACT", "true")
    app.config["ENABLE_EASYOCR"] = _env_flag("ENABLE_EASYOCR", "false")
    app.config["ENABLE_TROCR"] = _env_flag("ENABLE_TROCR", "false")
    app.config["OLLAMA_STATUS_TIMEOUT"] = float(os.environ.get("OLLAMA_STATUS_TIMEOUT", "1.0"))
    if app.config["AUTH_REQUIRED"] and not app.config["ADMIN_PASSWORD_HASH"]:
        raise RuntimeError("ADMIN_PASSWORD_HASH must be set when AUTH_REQUIRED=true")
