import copy
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .files import cleanup_files
from .services import process_ocr_file
from .storage import prune_old_runs, save_run


logger = logging.getLogger(__name__)


class OCRJobManager:
    def __init__(self, *, config, ocr_engine):
        self._config = {
            "UPLOAD_FOLDER": config["UPLOAD_FOLDER"],
            "PDF_FONT_PATH": config["PDF_FONT_PATH"],
            "DATABASE_PATH": config["DATABASE_PATH"],
            "MAX_SAVED_RUNS": config["MAX_SAVED_RUNS"],
            "OLLAMA_URL": config["OLLAMA_URL"],
            "OLLAMA_MODEL": config["OLLAMA_MODEL"],
            "OLLAMA_STATUS_TIMEOUT": config["OLLAMA_STATUS_TIMEOUT"],
        }
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
                "paths": copy.deepcopy(paths),
                "result": None,
            }
        self._executor.submit(self._run_job, job_id, copy.deepcopy(paths), copy.deepcopy(system_status))
        return self.snapshot(job_id)

    def snapshot(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def summary(self):
        with self._lock:
            counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            for job in self._jobs.values():
                status = job["status"]
                counts[status] = counts.get(status, 0) + 1
            counts["total"] = len(self._jobs)
            return counts

    def _run_job(self, job_id, paths, system_status):
        self._update(job_id, status="running", started_at=_utcnow())
        try:
            result = process_ocr_file(
                ocr_engine=self._ocr_engine,
                config=self._config,
                paths=paths,
                system_status=system_status,
            )
            run_id = save_run(
                self._config["DATABASE_PATH"],
                result["result_payload"],
                paths["pdf_path"],
                paths["preview_path"],
                result["overlay_data"].get("overlay_path"),
            )
            pruned = _prune_saved_runs(self._config["DATABASE_PATH"], self._config["UPLOAD_FOLDER"], self._config["MAX_SAVED_RUNS"])
            self._update(
                job_id,
                status="completed",
                completed_at=_utcnow(),
                result={
                    "run_id": run_id,
                    "payload": result["result_payload"],
                    "pdf_file_name": os.path.basename(paths["pdf_path"]),
                    "preview_file_name": os.path.basename(paths["preview_path"]),
                    "overlay_file_name": os.path.basename(paths["overlay_path"]) if result["overlay_data"].get("overlay_path") else "",
                    "pruned_runs": pruned,
                },
            )
        except Exception as exc:  # pragma: no cover - verified through route response
            logger.exception("Background OCR job %s failed", job_id)
            cleanup_files(paths["preview_path"], paths["overlay_path"], paths["pdf_path"])
            self._update(job_id, status="failed", completed_at=_utcnow(), error=str(exc))
        finally:
            cleanup_files(paths["file_path"], paths["preprocessed_path"])

    def _update(self, job_id, **changes):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(changes)


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


def _utcnow():
    return datetime.now(timezone.utc).isoformat()
