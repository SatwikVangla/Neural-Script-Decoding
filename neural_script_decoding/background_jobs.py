import copy
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .files import cleanup_files
from .ocr import MultiOCREngine
from .services import process_ocr_file
from .storage import prune_old_runs, save_run

try:  # pragma: no cover - optional dependency path
    from redis import Redis
except ImportError:  # pragma: no cover - optional dependency path
    Redis = None

try:  # pragma: no cover - optional dependency path
    from rq import Queue
    from rq.job import Job
except ImportError:  # pragma: no cover - optional dependency path
    Queue = None
    Job = None


logger = logging.getLogger(__name__)


def create_job_manager(*, config, ocr_engine):
    if config["OCR_QUEUE_BACKEND"] == "redis":
        return RedisOCRJobManager(config=config)
    return InProcessOCRJobManager(config=config, ocr_engine=ocr_engine)


class InProcessOCRJobManager:
    def __init__(self, *, config, ocr_engine):
        self._config = _job_config_snapshot(config)
        self._ocr_engine = ocr_engine
        self._executor = ThreadPoolExecutor(max_workers=config["BACKGROUND_OCR_WORKERS"], thread_name_prefix="ocr-job")
        self._jobs = {}
        self._lock = threading.Lock()

    def submit(self, *, paths, system_status):
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "created_at": _utcnow(),
                "started_at": None,
                "completed_at": None,
                "error": None,
                "result": None,
            }
        self._executor.submit(self._run_job, job_id, copy.deepcopy(paths), copy.deepcopy(system_status))
        return self.snapshot(job_id)

    def snapshot(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            snapshot = copy.deepcopy(job)
        return snapshot

    def summary(self):
        with self._lock:
            counts = {"backend": "local", "queued": 0, "running": 0, "completed": 0, "failed": 0}
            for job in self._jobs.values():
                counts[job["status"]] = counts.get(job["status"], 0) + 1
            counts["total"] = len(self._jobs)
            return counts

    def _run_job(self, job_id, paths, system_status):
        self._update(job_id, status="running", started_at=_utcnow())
        try:
            result = run_ocr_job(paths=paths, system_status=system_status, config_snapshot=self._config, ocr_engine=self._ocr_engine)
            self._update(job_id, status="completed", completed_at=_utcnow(), result=result)
        except Exception as exc:  # pragma: no cover - verified through route response
            logger.exception("Background OCR job %s failed", job_id)
            self._update(job_id, status="failed", completed_at=_utcnow(), error=str(exc))

    def _update(self, job_id, **changes):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)


class RedisOCRJobManager:
    def __init__(self, *, config):
        if Redis is None or Queue is None or Job is None:
            raise RuntimeError("Redis queue backend requires the 'redis' and 'rq' packages")
        self._config = _job_config_snapshot(config)
        self._connection = Redis.from_url(config["REDIS_URL"])
        self._queue = Queue(config["OCR_QUEUE_NAME"], connection=self._connection, default_timeout=config["BACKGROUND_JOB_TIMEOUT"])

    def submit(self, *, paths, system_status):
        job = self._queue.enqueue(
            run_ocr_job,
            kwargs={
                "paths": copy.deepcopy(paths),
                "system_status": copy.deepcopy(system_status),
                "config_snapshot": self._config,
            },
            job_timeout=self._config["BACKGROUND_JOB_TIMEOUT"],
        )
        return self.snapshot(job.id)

    def snapshot(self, job_id):
        try:
            job = Job.fetch(job_id, connection=self._connection)
        except Exception:
            return None

        status = _rq_status(job.get_status(refresh=True))
        snapshot = {
            "id": job.id,
            "status": status,
            "created_at": _to_iso(job.created_at),
            "started_at": _to_iso(job.started_at),
            "completed_at": _to_iso(job.ended_at),
            "error": job.exc_info.splitlines()[-1] if job.exc_info else None,
            "result": job.result if status == "completed" else None,
        }
        return snapshot

    def summary(self):
        return {
            "backend": "redis",
            "queued": len(self._queue.job_ids),
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": len(self._queue.job_ids),
            "queue_name": self._queue.name,
        }


def run_ocr_job(*, paths, system_status, config_snapshot, ocr_engine=None):
    engine = ocr_engine or MultiOCREngine(config_snapshot)
    engine.initialize_engines()
    result = None
    try:
        result = process_ocr_file(
            ocr_engine=engine,
            config=config_snapshot,
            paths=paths,
            system_status=system_status,
        )
        run_id = save_run(
            config_snapshot["DATABASE_PATH"],
            result["result_payload"],
            paths["pdf_path"],
            paths["preview_path"],
            result["overlay_data"].get("overlay_path"),
        )
        pruned = _prune_saved_runs(
            config_snapshot["DATABASE_PATH"],
            config_snapshot["UPLOAD_FOLDER"],
            config_snapshot["MAX_SAVED_RUNS"],
        )
        return {
            "run_id": run_id,
            "payload": result["result_payload"],
            "pdf_file_name": os.path.basename(paths["pdf_path"]),
            "preview_file_name": os.path.basename(paths["preview_path"]),
            "overlay_file_name": os.path.basename(paths["overlay_path"]) if result["overlay_data"].get("overlay_path") else "",
            "pruned_runs": pruned,
        }
    except Exception:
        cleanup_files(paths["preview_path"], paths["overlay_path"], paths["pdf_path"])
        raise
    finally:
        cleanup_files(paths["file_path"], paths["preprocessed_path"])


def _job_config_snapshot(config):
    return {
        "UPLOAD_FOLDER": config["UPLOAD_FOLDER"],
        "PDF_FONT_PATH": config["PDF_FONT_PATH"],
        "DATABASE_PATH": config["DATABASE_PATH"],
        "MAX_SAVED_RUNS": config["MAX_SAVED_RUNS"],
        "OLLAMA_URL": config["OLLAMA_URL"],
        "OLLAMA_MODEL": config["OLLAMA_MODEL"],
        "OLLAMA_STATUS_TIMEOUT": config["OLLAMA_STATUS_TIMEOUT"],
        "BACKGROUND_JOB_TIMEOUT": config["BACKGROUND_JOB_TIMEOUT"],
        "OCR_QUEUE_NAME": config["OCR_QUEUE_NAME"],
        "REDIS_URL": config["REDIS_URL"],
    }


def _prune_saved_runs(database_path, upload_folder, keep_limit):
    deleted = prune_old_runs(database_path, keep_limit)
    for run in deleted:
        if run.get("pdf_file_name"):
            cleanup_files(os.path.join(upload_folder, run["pdf_file_name"]))
        if run.get("preview_file_name"):
            cleanup_files(os.path.join(upload_folder, run["preview_file_name"]))
        if run.get("overlay_file_name"):
            cleanup_files(os.path.join(upload_folder, run["overlay_file_name"]))
    return [dict(run) for run in deleted]


def _rq_status(status):
    return {
        "queued": "queued",
        "started": "running",
        "finished": "completed",
        "failed": "failed",
        "deferred": "queued",
        "scheduled": "queued",
    }.get(status, status)


def _to_iso(value):
    if value is None:
        return None
    return value.isoformat()


def _utcnow():
    return datetime.now(timezone.utc).isoformat()
