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
from .storage import create_job, get_job, list_jobs, prune_jobs, prune_old_runs, save_run, summarize_jobs, update_job

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
WORKER_HEARTBEAT_KEY = "ocr-worker-heartbeat"


def worker_heartbeat_key(queue_name):
    return f"{WORKER_HEARTBEAT_KEY}:{queue_name}"


def create_job_manager(*, config, ocr_engine):
    if config["OCR_QUEUE_BACKEND"] == "redis":
        return RedisOCRJobManager(config=config)
    return InProcessOCRJobManager(config=config, ocr_engine=ocr_engine)


class InProcessOCRJobManager:
    def __init__(self, *, config, ocr_engine):
        self._config = _job_config_snapshot(config)
        self._ocr_engine = ocr_engine
        self._executor = ThreadPoolExecutor(max_workers=config["BACKGROUND_OCR_WORKERS"], thread_name_prefix="ocr-job")
        self._lock = threading.Lock()
        self._futures = {}

    def submit(self, *, paths, system_status, job_id=None, attempt_count=None):
        job_id = job_id or uuid.uuid4().hex
        if attempt_count is None:
            create_job(
                self._config["DATABASE_PATH"],
                job_id=job_id,
                paths=paths,
                system_status=system_status,
                backend="local",
            )
        else:
            update_job(
                self._config["DATABASE_PATH"],
                job_id,
                status="queued",
                attempt_count=attempt_count,
                error_message=None,
                result_payload_json=None,
                run_id=None,
                pdf_file_name=None,
                preview_file_name=None,
                overlay_file_name=None,
                started_at=None,
                completed_at=None,
                canceled_at=None,
                paths=paths,
                system_status=system_status,
            )

        future = self._executor.submit(self._run_job, job_id, copy.deepcopy(paths), copy.deepcopy(system_status))
        with self._lock:
            self._futures[job_id] = future
        return self.snapshot(job_id)

    def snapshot(self, job_id):
        return get_job(self._config["DATABASE_PATH"], job_id)

    def summary(self):
        self.prune_finished()
        counts = summarize_jobs(self._config["DATABASE_PATH"])
        counts["backend"] = "local"
        counts["worker_healthy"] = True
        return counts

    def list(self, limit=50):
        return list_jobs(self._config["DATABASE_PATH"], limit=limit)

    def cancel(self, job_id):
        with self._lock:
            future = self._futures.get(job_id)
        snapshot = self.snapshot(job_id)
        if snapshot is None:
            return None
        if snapshot["status"] != "queued" or future is None or not future.cancel():
            return snapshot
        update_job(
            self._config["DATABASE_PATH"],
            job_id,
            status="canceled",
            error_message="Canceled before execution",
            canceled_at=_utcnow(),
            completed_at=_utcnow(),
        )
        return self.snapshot(job_id)

    def retry(self, job_id):
        job = self.snapshot(job_id)
        if job is None:
            return None
        if job["status"] not in {"failed", "canceled"}:
            return job
        if not os.path.exists(job["paths"]["file_path"]):
            update_job(
                self._config["DATABASE_PATH"],
                job_id,
                status="failed",
                error_message="Original upload is no longer available for retry",
                completed_at=_utcnow(),
            )
            return self.snapshot(job_id)
        return self.submit(
            job_id=job_id,
            paths=job["paths"],
            system_status=job["system_status"],
            attempt_count=(job["attempt_count"] or 0) + 1,
        )

    def prune_finished(self):
        deleted_jobs = prune_jobs(
            self._config["DATABASE_PATH"],
            keep_limit=self._config["MAX_STORED_JOBS"],
            retention_days=self._config["JOB_RETENTION_DAYS"],
        )
        _cleanup_job_artifacts(deleted_jobs)
        return deleted_jobs

    def worker_health(self):
        return {"healthy": True, "reason": "In-process workers active"}

    def _run_job(self, job_id, paths, system_status):
        try:
            run_ocr_job(
                job_id=job_id,
                paths=paths,
                system_status=system_status,
                config_snapshot=self._config,
                ocr_engine=self._ocr_engine,
            )
        except Exception:  # pragma: no cover - verified through route response
            logger.exception("Background OCR job %s failed", job_id)
        finally:
            with self._lock:
                self._futures.pop(job_id, None)


class RedisOCRJobManager:
    def __init__(self, *, config):
        if Redis is None or Queue is None or Job is None:
            raise RuntimeError("Redis queue backend requires the 'redis' and 'rq' packages")
        self._config = _job_config_snapshot(config)
        self._connection = Redis.from_url(config["REDIS_URL"])
        self._queue = Queue(config["OCR_QUEUE_NAME"], connection=self._connection, default_timeout=config["BACKGROUND_JOB_TIMEOUT"])

    def submit(self, *, paths, system_status, job_id=None, attempt_count=None):
        job_id = job_id or uuid.uuid4().hex
        job = self._queue.enqueue(
            run_ocr_job,
            kwargs={
                "job_id": job_id,
                "paths": copy.deepcopy(paths),
                "system_status": copy.deepcopy(system_status),
                "config_snapshot": self._config,
            },
            job_timeout=self._config["BACKGROUND_JOB_TIMEOUT"],
        )
        if attempt_count is None:
            create_job(
                self._config["DATABASE_PATH"],
                job_id=job_id,
                paths=paths,
                system_status=system_status,
                backend="redis",
                queue_job_id=job.id,
            )
        else:
            update_job(
                self._config["DATABASE_PATH"],
                job_id,
                status="queued",
                attempt_count=attempt_count,
                queue_job_id=job.id,
                error_message=None,
                result_payload_json=None,
                run_id=None,
                pdf_file_name=None,
                preview_file_name=None,
                overlay_file_name=None,
                started_at=None,
                completed_at=None,
                canceled_at=None,
                paths=paths,
                system_status=system_status,
            )
        return self.snapshot(job_id)

    def snapshot(self, job_id):
        snapshot = get_job(self._config["DATABASE_PATH"], job_id)
        if snapshot is None:
            return None
        if snapshot.get("queue_job_id") and snapshot["status"] in {"queued", "running"}:
            try:
                job = Job.fetch(snapshot["queue_job_id"], connection=self._connection)
            except Exception:
                return snapshot
            status = _rq_status(job.get_status(refresh=True))
            if status != snapshot["status"]:
                snapshot = update_job(
                    self._config["DATABASE_PATH"],
                    job_id,
                    status=status,
                    started_at=_to_iso(job.started_at) or snapshot["started_at"],
                    completed_at=_to_iso(job.ended_at) or snapshot["completed_at"],
                    error_message=job.exc_info.splitlines()[-1] if job.exc_info else snapshot["error_message"],
                )
        return snapshot

    def summary(self):
        self.prune_finished()
        counts = summarize_jobs(self._config["DATABASE_PATH"])
        counts["backend"] = "redis"
        counts["queue_name"] = self._queue.name
        counts["worker_healthy"] = self.worker_health()["healthy"]
        return counts

    def list(self, limit=50):
        return list_jobs(self._config["DATABASE_PATH"], limit=limit)

    def cancel(self, job_id):
        snapshot = self.snapshot(job_id)
        if snapshot is None:
            return None
        if snapshot["status"] != "queued" or not snapshot.get("queue_job_id"):
            return snapshot
        try:
            job = Job.fetch(snapshot["queue_job_id"], connection=self._connection)
            job.cancel()
        except Exception:
            return snapshot
        return update_job(
            self._config["DATABASE_PATH"],
            job_id,
            status="canceled",
            error_message="Canceled before execution",
            canceled_at=_utcnow(),
            completed_at=_utcnow(),
        )

    def retry(self, job_id):
        job = self.snapshot(job_id)
        if job is None:
            return None
        if job["status"] not in {"failed", "canceled"}:
            return job
        if not os.path.exists(job["paths"]["file_path"]):
            return update_job(
                self._config["DATABASE_PATH"],
                job_id,
                status="failed",
                error_message="Original upload is no longer available for retry",
                completed_at=_utcnow(),
            )
        return self.submit(
            job_id=job_id,
            paths=job["paths"],
            system_status=job["system_status"],
            attempt_count=(job["attempt_count"] or 0) + 1,
        )

    def prune_finished(self):
        deleted_jobs = prune_jobs(
            self._config["DATABASE_PATH"],
            keep_limit=self._config["MAX_STORED_JOBS"],
            retention_days=self._config["JOB_RETENTION_DAYS"],
        )
        _cleanup_job_artifacts(deleted_jobs)
        return deleted_jobs

    def worker_health(self):
        heartbeat = None
        try:
            heartbeat = self._connection.get(worker_heartbeat_key(self._queue.name))
        except Exception:
            return {"healthy": False, "reason": "Unable to read worker heartbeat"}
        if not heartbeat:
            return {"healthy": False, "reason": "No worker heartbeat reported"}
        heartbeat_value = heartbeat.decode("utf-8") if isinstance(heartbeat, bytes) else str(heartbeat)
        return {"healthy": True, "reason": "Heartbeat active", "last_seen": heartbeat_value}


def run_ocr_job(*, job_id, paths, system_status, config_snapshot, ocr_engine=None):
    engine = ocr_engine or MultiOCREngine(config_snapshot)
    engine.initialize_engines()
    update_job(
        config_snapshot["DATABASE_PATH"],
        job_id,
        status="running",
        started_at=_utcnow(),
        completed_at=None,
        canceled_at=None,
        error_message=None,
    )
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
        cleanup_files(paths["file_path"], paths["preprocessed_path"])
        update_job(
            config_snapshot["DATABASE_PATH"],
            job_id,
            status="completed",
            completed_at=_utcnow(),
            error_message=None,
            result_payload=result["result_payload"],
            run_id=run_id,
            pdf_file_name=os.path.basename(paths["pdf_path"]),
            preview_file_name=os.path.basename(paths["preview_path"]),
            overlay_file_name=os.path.basename(paths["overlay_path"]) if result["overlay_data"].get("overlay_path") else "",
        )
        return {
            "run_id": run_id,
            "payload": result["result_payload"],
            "pdf_file_name": os.path.basename(paths["pdf_path"]),
            "preview_file_name": os.path.basename(paths["preview_path"]),
            "overlay_file_name": os.path.basename(paths["overlay_path"]) if result["overlay_data"].get("overlay_path") else "",
            "pruned_runs": pruned,
        }
    except Exception as exc:
        cleanup_files(paths["preview_path"], paths["overlay_path"], paths["pdf_path"])
        update_job(
            config_snapshot["DATABASE_PATH"],
            job_id,
            status="failed",
            completed_at=_utcnow(),
            error_message=str(exc),
        )
        raise
    finally:
        cleanup_files(paths["preprocessed_path"])


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
        "MAX_STORED_JOBS": config["MAX_STORED_JOBS"],
        "JOB_RETENTION_DAYS": config["JOB_RETENTION_DAYS"],
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


def _cleanup_job_artifacts(jobs):
    for job in jobs:
        paths = job.get("paths", {})
        cleanup_files(paths.get("file_path"), paths.get("preprocessed_path"))
        if not job.get("run_id"):
            cleanup_files(
                job.get("preview_file_name") and os.path.join(os.path.dirname(paths.get("file_path", "")), job["preview_file_name"]),
                job.get("overlay_file_name") and os.path.join(os.path.dirname(paths.get("file_path", "")), job["overlay_file_name"]),
                job.get("pdf_file_name") and os.path.join(os.path.dirname(paths.get("file_path", "")), job["pdf_file_name"]),
            )


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
