from waitress import serve

from app import app


if __name__ == "__main__":
    serve(
        app,
        host=app.config["APP_HOST"],
        port=app.config["APP_PORT"],
        threads=4,
    )
