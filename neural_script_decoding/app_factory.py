import logging
import threading
from pathlib import Path

from flask import Flask

from .background_jobs import create_job_manager
from .config import apply_config
from .ocr import MultiOCREngine
from .services import warmup_ollama
from .storage import init_db, upsert_user
from .web import register_routes


def create_app():
    logging.basicConfig(level=logging.INFO)
    package_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str((package_dir.parent / "templates").resolve()),
        static_folder=str((package_dir.parent / "static").resolve()),
    )
    apply_config(app)
    logging.getLogger().setLevel(app.config["LOG_LEVEL"])
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    init_db(app.config["DATABASE_PATH"])
    if app.config.get("ADMIN_PASSWORD_HASH"):
        upsert_user(
            app.config["DATABASE_PATH"],
            username=app.config["ADMIN_USERNAME"],
            password_hash=app.config["ADMIN_PASSWORD_HASH"],
            is_active=True,
            role="admin",
        )
    app.extensions["ocr_engine"] = MultiOCREngine(app.config)
    app.extensions["ocr_job_manager"] = create_job_manager(config=app.config, ocr_engine=app.extensions["ocr_engine"])
    if app.config.get("OLLAMA_WARMUP_ENABLED", True):
        threading.Thread(target=warmup_ollama, args=(app.config,), name="ollama-warmup", daemon=True).start()
    register_routes(app)
    return app
