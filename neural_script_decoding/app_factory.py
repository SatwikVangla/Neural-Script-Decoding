import logging
from pathlib import Path

from flask import Flask

from .background_jobs import OCRJobManager
from .config import apply_config
from .ocr import MultiOCREngine
from .storage import init_db
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
    app.extensions["ocr_engine"] = MultiOCREngine(app.config)
    app.extensions["ocr_job_manager"] = OCRJobManager(config=app.config, ocr_engine=app.extensions["ocr_engine"])
    register_routes(app)
    return app
