import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask
from redis import Redis
from rq import Connection, Worker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_script_decoding.background_jobs import worker_heartbeat_key
from neural_script_decoding.config import apply_config


def load_config():
    app = Flask(__name__)
    apply_config(app)
    return app.config


def main():
    config = load_config()
    connection = Redis.from_url(config["REDIS_URL"])
    heartbeat_key = worker_heartbeat_key(config["OCR_QUEUE_NAME"])
    stop_event = threading.Event()

    def publish_heartbeat():
        while not stop_event.is_set():
            connection.set(
                heartbeat_key,
                datetime.now(timezone.utc).isoformat(),
                ex=max(config["BACKGROUND_JOB_TIMEOUT"], 30),
            )
            stop_event.wait(5)

    heartbeat_thread = threading.Thread(target=publish_heartbeat, name="ocr-worker-heartbeat", daemon=True)
    heartbeat_thread.start()
    with Connection(connection):
        worker = Worker([config["OCR_QUEUE_NAME"]])
        try:
            worker.work()
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=1)


if __name__ == "__main__":
    main()
