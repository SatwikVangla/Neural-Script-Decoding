import os


def apply_config(app):
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))
    app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "static/uploads")
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "16")) * 1024 * 1024
    app.config["OLLAMA_URL"] = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    app.config["OLLAMA_MODEL"] = os.environ.get("OLLAMA_MODEL", "mistral")
    app.config["PDF_FONT_PATH"] = os.environ.get("PDF_FONT_PATH", os.path.join("static", "DejaVuSans.ttf"))
    app.config["DATABASE_PATH"] = os.environ.get("DATABASE_PATH", os.path.join("instance", "neural_script_decoding.sqlite3"))
    app.config["APP_HOST"] = os.environ.get("APP_HOST", "0.0.0.0")
    app.config["APP_PORT"] = int(os.environ.get("APP_PORT", "5000"))
    app.config["APP_DEBUG"] = os.environ.get("APP_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    app.config["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "INFO").upper()
    app.config["ENABLE_TESSERACT"] = os.environ.get("ENABLE_TESSERACT", "true").lower() in {"1", "true", "yes", "on"}
    app.config["ENABLE_EASYOCR"] = os.environ.get("ENABLE_EASYOCR", "false").lower() in {"1", "true", "yes", "on"}
    app.config["ENABLE_TROCR"] = os.environ.get("ENABLE_TROCR", "false").lower() in {"1", "true", "yes", "on"}
    app.config["OLLAMA_STATUS_TIMEOUT"] = float(os.environ.get("OLLAMA_STATUS_TIMEOUT", "1.0"))
