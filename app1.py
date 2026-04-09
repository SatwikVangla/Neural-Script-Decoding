from app import app


ocr_engine = app.extensions["ocr_engine"]


if __name__ == "__main__":
    app.run(
        debug=app.config["APP_DEBUG"],
        host=app.config["APP_HOST"],
        port=app.config["APP_PORT"],
    )
