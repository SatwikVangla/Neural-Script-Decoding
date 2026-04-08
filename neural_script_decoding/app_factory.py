import logging
import os

from flask import Flask

from .config import apply_config
from .ocr import MultiOCREngine
from .storage import init_db
from .web import register_routes


def create_app():
    logging.basicConfig(level=logging.INFO)
    package_dir = os.path.dirname(__file__)
    app = Flask(
        __name__,
        template_folder=os.path.join(package_dir, "..", "templates"),
        static_folder=os.path.join(package_dir, "..", "static"),
    )
    apply_config(app)
    logging.getLogger().setLevel(app.config["LOG_LEVEL"])
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    init_db(app.config["DATABASE_PATH"])
    app.extensions["ocr_engine"] = MultiOCREngine(app.config)
    register_routes(app)
    return app
