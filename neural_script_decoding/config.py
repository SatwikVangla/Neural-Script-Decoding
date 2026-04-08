import os


def apply_config(app):
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))
    app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "static/uploads")
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "16")) * 1024 * 1024
    app.config["OLLAMA_URL"] = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    app.config["OLLAMA_MODEL"] = os.environ.get("OLLAMA_MODEL", "mistral")
    app.config["PDF_FONT_PATH"] = os.environ.get("PDF_FONT_PATH", os.path.join("static", "DejaVuSans.ttf"))
