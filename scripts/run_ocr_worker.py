from flask import Flask
from redis import Redis
from rq import Connection, Worker

from neural_script_decoding.config import apply_config


def load_config():
    app = Flask(__name__)
    apply_config(app)
    return app.config


def main():
    config = load_config()
    connection = Redis.from_url(config["REDIS_URL"])
    with Connection(connection):
        worker = Worker([config["OCR_QUEUE_NAME"]])
        worker.work()


if __name__ == "__main__":
    main()
