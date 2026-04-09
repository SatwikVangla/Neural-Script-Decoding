import logging
import os
import json
import time
from datetime import datetime
from functools import wraps

from flask import abort, current_app, flash, jsonify, redirect, render_template, request, send_from_directory, url_for

from .files import allowed_file, build_upload_paths, cleanup_files
from .services import diagnostics_snapshot, json_download_response, ollama_status, process_ocr_file
from .storage import delete_run, get_run, list_runs, prune_old_runs, save_run


logger = logging.getLogger(__name__)
_RATE_LIMIT_STATE = {}


def register_routes(app):
    @app.route("/")
    def index():
        return render_template("index.html", system_status=_system_status())

    @app.route("/history")
    def history():
        query = request.args.get("q", "").strip()
        engine = request.args.get("engine", "").strip() or None
        runs = list_runs(
            current_app.config["DATABASE_PATH"],
            query=query or None,
            engine=engine,
        )
        return render_template(
            "history.html",
            runs=runs,
            system_status=_system_status(),
            search_query=query,
            selected_engine=engine or "",
        )

    @app.route("/jobs")
    def jobs():
        return render_template(
            "jobs.html",
            jobs=_job_manager().list(limit=100),
            summary=_job_manager().summary(),
            system_status=_system_status(),
        )

    @app.route("/system")
    def system_admin():
        return render_template(
            "system.html",
            operations=_operations_snapshot(),
            system_status=_system_status(),
        )

    @app.route("/jobs/<job_id>")
    def job_detail(job_id):
        job = _job_manager().snapshot(job_id)
        if job is None:
            abort(404)
        return render_template(
            "job_detail.html",
            job=job,
            system_status=_system_status(),
        )

    @app.route("/history/<int:run_id>")
    def history_detail(run_id):
        run = get_run(current_app.config["DATABASE_PATH"], run_id)
        if run is None:
            abort(404)
        payload = run["payload"]
        return render_template(
            "history_detail.html",
            run=run,
            ocr_results=payload.get("ocr_results", []),
            raw_text=payload.get("raw_text", ""),
            corrected_text=payload.get("corrected_text", ""),
            correction=payload.get("correction", {}),
            pdf_path=os.path.join(current_app.config["UPLOAD_FOLDER"], run["pdf_file_name"]) if run.get("pdf_file_name") else "",
            preview_url=url_for("preview_image", filename=run["preview_file_name"]) if run.get("preview_file_name") else "",
            overlay_url=url_for("overlay_image", filename=run["overlay_file_name"]) if run.get("overlay_file_name") else "",
            regions=payload.get("regions", []),
            system_status=payload.get("system", _system_status()),
            result_payload=payload,
        )

    @app.route("/history/<int:run_id>/delete", methods=["POST"])
    def delete_history_run(run_id):
        deleted = delete_run(current_app.config["DATABASE_PATH"], run_id)
        if deleted is None:
            abort(404)
        if deleted.get("pdf_file_name"):
            cleanup_files(os.path.join(current_app.config["UPLOAD_FOLDER"], deleted["pdf_file_name"]))
        if deleted.get("preview_file_name"):
            cleanup_files(os.path.join(current_app.config["UPLOAD_FOLDER"], deleted["preview_file_name"]))
        if deleted.get("overlay_file_name"):
            cleanup_files(os.path.join(current_app.config["UPLOAD_FOLDER"], deleted["overlay_file_name"]))
        flash("Saved run deleted")
        logger.info("Deleted saved OCR run %s", run_id)
        return redirect(url_for("history"))

    @app.route("/upload", methods=["POST"])
    def upload_file():
        if "file" not in request.files:
            flash("No file part")
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(url_for("index"))

        if file and allowed_file(file.filename):
            paths = build_upload_paths(current_app.config["UPLOAD_FOLDER"], file.filename)
            file.save(paths["file_path"])

            try:
                logger.info("Processing image: %s", paths["filename"])
                result = _run_ocr_pipeline(paths)
                cleanup_files(paths["file_path"], paths["preprocessed_path"])
                run_id = save_run(
                    current_app.config["DATABASE_PATH"],
                    result["result_payload"],
                    paths["pdf_path"],
                    paths["preview_path"],
                    result["overlay_data"].get("overlay_path"),
                )
                _prune_saved_runs()
                return render_template(
                    "results.html",
                    ocr_results=result["ocr_results"],
                    raw_text=result["raw_text"],
                    corrected_text=result["corrected_text"],
                    correction=result["correction"],
                    pdf_path=paths["pdf_path"],
                    preview_url=url_for("preview_image", filename=os.path.basename(paths["preview_path"])),
                    overlay_url=url_for("overlay_image", filename=os.path.basename(paths["overlay_path"])) if result["overlay_data"].get("overlay_path") else "",
                    regions=result["overlay_data"].get("regions", []),
                    system_status=_system_status(),
                    result_payload=result["result_payload"],
                    run_id=run_id,
                )
            except Exception as exc:
                logger.exception("Processing failed")
                flash(f"Error processing image: {exc}")
                cleanup_files(paths["file_path"], paths["preprocessed_path"], paths["preview_path"], paths["overlay_path"])
                return redirect(url_for("index"))

        flash("Allowed file types are png, jpg, jpeg, gif, bmp")
        return redirect(url_for("index"))

    @app.route("/download/<filename>")
    def download_pdf(filename):
        safe_name = os.path.basename(filename)
        if safe_name != filename:
            abort(404)
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], safe_name)
        if os.path.exists(file_path):
            logger.info("Serving generated PDF %s", safe_name)
            return send_from_directory(current_app.config["UPLOAD_FOLDER"], safe_name, as_attachment=True)
        flash("File not found")
        return redirect(url_for("index"))

    @app.route("/api/ocr", methods=["POST"])
    @require_api_access
    def api_ocr():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        paths = None
        try:
            paths = build_upload_paths(current_app.config["UPLOAD_FOLDER"], file.filename)
            file.save(paths["file_path"])
            logger.info("API OCR request for %s", paths["filename"])
            result = _run_ocr_pipeline(paths)
            run_id = save_run(
                current_app.config["DATABASE_PATH"],
                result["result_payload"],
                paths["pdf_path"],
                paths["preview_path"],
                result["overlay_data"].get("overlay_path"),
            )
            _prune_saved_runs()
            return jsonify(
                {
                    "success": True,
                    "run_id": run_id,
                    "pdf_url": url_for("download_pdf", filename=os.path.basename(paths["pdf_path"])),
                    "preview_url": url_for("preview_image", filename=os.path.basename(paths["preview_path"])),
                    "overlay_url": url_for("overlay_image", filename=os.path.basename(paths["overlay_path"])) if result["overlay_data"].get("overlay_path") else "",
                    **result["result_payload"],
                }
            )
        except Exception as exc:
            logger.exception("API OCR failed")
            return jsonify({"error": str(exc)}), 500
        finally:
            if paths:
                cleanup_files(paths["file_path"], paths["preprocessed_path"])

    @app.route("/api/ocr/jobs", methods=["POST"])
    @require_api_access
    def api_ocr_async():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        paths = build_upload_paths(current_app.config["UPLOAD_FOLDER"], file.filename)
        try:
            file.save(paths["file_path"])
            logger.info("Queued async OCR request for %s", paths["filename"])
            job = _job_manager().submit(paths=paths, system_status=_system_status())
            return (
                jsonify(
                    {
                        "success": True,
                        "job_id": job["id"],
                        "status": job["status"],
                        "status_url": url_for("api_ocr_job_status", job_id=job["id"]),
                        "created_at": job["created_at"],
                    }
                ),
                202,
            )
        except Exception as exc:
            logger.exception("Async OCR queue submission failed")
            cleanup_files(paths["file_path"], paths["preprocessed_path"], paths["preview_path"], paths["overlay_path"])
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/ocr/jobs/<job_id>", methods=["GET"])
    @require_api_access
    def api_ocr_job_status(job_id):
        job = _job_manager().snapshot(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404

        response = {
            "job_id": job["id"],
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
            "error": job["error_message"],
            "attempt_count": job["attempt_count"],
            "run_id": job["run_id"],
        }
        if job["result_payload"]:
            response.update(
                {
                    "pdf_url": url_for("download_pdf", filename=job["pdf_file_name"]),
                    "preview_url": url_for("preview_image", filename=job["preview_file_name"]),
                    "overlay_url": url_for("overlay_image", filename=job["overlay_file_name"]) if job["overlay_file_name"] else "",
                    **job["result_payload"],
                }
            )
        return jsonify(response)

    @app.route("/api/ocr/jobs/<job_id>/cancel", methods=["POST"])
    @require_api_access
    def api_ocr_job_cancel(job_id):
        job = _job_manager().cancel(job_id)
        if job is None:
            return jsonify({"error": "Job not found"}), 404
        if job["status"] != "canceled":
            return jsonify({"error": f"Job cannot be canceled from status {job['status']}", "status": job["status"]}), 409
        return jsonify({"success": True, "job_id": job_id, "status": job["status"]})

    @app.route("/api/ocr/jobs/<job_id>/retry", methods=["POST"])
    @require_api_access
    def api_ocr_job_retry(job_id):
        before = _job_manager().snapshot(job_id)
        if before is None:
            return jsonify({"error": "Job not found"}), 404
        job = _job_manager().retry(job_id)
        if job["status"] not in {"queued", "running"}:
            return jsonify({"error": f"Job cannot be retried from status {before['status']}", "status": job["status"]}), 409
        return jsonify(
            {
                "success": True,
                "job_id": job_id,
                "status": job["status"],
                "attempt_count": job["attempt_count"],
                "status_url": url_for("api_ocr_job_status", job_id=job_id),
            }
        )

    @app.route("/jobs/<job_id>/cancel", methods=["POST"])
    def cancel_job(job_id):
        job = _job_manager().cancel(job_id)
        if job is None:
            abort(404)
        if job["status"] == "canceled":
            flash("Job canceled")
        else:
            flash(f"Job could not be canceled from status {job['status']}")
        return redirect(request.referrer or url_for("job_detail", job_id=job_id))

    @app.route("/jobs/<job_id>/retry", methods=["POST"])
    def retry_job(job_id):
        job = _job_manager().retry(job_id)
        if job is None:
            abort(404)
        if job["status"] in {"queued", "running"}:
            flash("Job re-queued")
        else:
            flash(f"Job could not be retried from status {job['status']}")
        return redirect(request.referrer or url_for("job_detail", job_id=job_id))

    @app.route("/health")
    def health_check():
        logger.info("Health check requested")
        return jsonify(
            {
                "status": "healthy",
                "initialized": _ocr_engine().initialized,
                "system": _system_status(),
                "background_jobs": _job_manager().summary(),
                "worker": _job_manager().worker_health(),
                "timestamp": datetime.now().isoformat(),
            }
        )

    @app.route("/diagnostics")
    def diagnostics():
        system_status = _system_status()
        return render_template(
            "diagnostics.html",
            diagnostics=diagnostics_snapshot(
                config=current_app.config,
                ocr_engine=_ocr_engine(),
                system_status=system_status,
            ),
            system_status=system_status,
        )

    @app.route("/download-json", methods=["POST"])
    def download_json():
        payload = request.get_json(silent=True)
        if payload is None:
            raw_payload = request.form.get("payload", "").strip()
            if raw_payload:
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    return jsonify({"error": "Invalid result payload"}), 400
        if not payload:
            return jsonify({"error": "No result payload provided"}), 400

        filename = payload.get("file", "ocr-result")
        stem = os.path.splitext(filename)[0] or "ocr-result"
        logger.info("Serving structured OCR export for %s", stem)
        return json_download_response(payload, f"{stem}.json")

    @app.route("/preview/<filename>")
    def preview_image(filename):
        safe_name = os.path.basename(filename)
        if safe_name != filename:
            abort(404)
        preview_path = os.path.join(current_app.config["UPLOAD_FOLDER"], safe_name)
        if os.path.exists(preview_path):
            return send_from_directory(current_app.config["UPLOAD_FOLDER"], safe_name)
        abort(404)

    @app.route("/overlay/<filename>")
    def overlay_image(filename):
        safe_name = os.path.basename(filename)
        if safe_name != filename:
            abort(404)
        overlay_path = os.path.join(current_app.config["UPLOAD_FOLDER"], safe_name)
        if os.path.exists(overlay_path):
            return send_from_directory(current_app.config["UPLOAD_FOLDER"], safe_name)
        abort(404)


def _ocr_engine():
    return current_app.extensions["ocr_engine"]


def _job_manager():
    return current_app.extensions["ocr_job_manager"]


def _system_status():
    _ocr_engine().initialize_engines()
    return {
        "engines": _ocr_engine().available_summary(),
        "ollama": ollama_status(current_app.config),
    }


def _run_ocr_pipeline(paths):
    return process_ocr_file(
        ocr_engine=_ocr_engine(),
        config=current_app.config,
        paths=paths,
        system_status=_system_status(),
    )


def _operations_snapshot():
    summary = _job_manager().summary()
    worker = _job_manager().worker_health()
    return {
        "api": {
            "auth_required": bool(current_app.config.get("API_KEY", "").strip()),
            "rate_limit": current_app.config.get("API_RATE_LIMIT", 0),
            "rate_window_seconds": current_app.config.get("API_RATE_WINDOW_SECONDS", 60),
        },
        "jobs": {
            "backend": summary.get("backend"),
            "queue_name": summary.get("queue_name", ""),
            "worker_healthy": worker.get("healthy", False),
            "worker_reason": worker.get("reason", "Unknown"),
            "worker_last_seen": worker.get("last_seen", ""),
            "queued": summary.get("queued", 0),
            "running": summary.get("running", 0),
            "completed": summary.get("completed", 0),
            "failed": summary.get("failed", 0),
            "canceled": summary.get("canceled", 0),
            "total": summary.get("total", 0),
            "max_stored_jobs": current_app.config.get("MAX_STORED_JOBS"),
            "retention_days": current_app.config.get("JOB_RETENTION_DAYS"),
        },
    }


def require_api_access(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        unauthorized = _validate_api_key()
        if unauthorized is not None:
            return unauthorized
        limited = _enforce_rate_limit()
        if limited is not None:
            return limited
        return view(*args, **kwargs)

    return wrapped


def _validate_api_key():
    api_key = current_app.config.get("API_KEY", "").strip()
    if not api_key:
        return None

    presented = request.headers.get("X-API-Key", "").strip()
    if not presented:
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            presented = authorization.split(" ", 1)[1].strip()
    if presented == api_key:
        return None
    return jsonify({"error": "Unauthorized"}), 401


def _enforce_rate_limit():
    limit = current_app.config.get("API_RATE_LIMIT", 0)
    window = current_app.config.get("API_RATE_WINDOW_SECONDS", 60)
    if limit <= 0:
        return None

    identity = request.headers.get("X-API-Key") or request.remote_addr or "anonymous"
    now = time.time()
    bucket = _RATE_LIMIT_STATE.setdefault(identity, [])
    bucket[:] = [timestamp for timestamp in bucket if now - timestamp < window]
    if len(bucket) >= limit:
        retry_after = max(1, int(window - (now - bucket[0])))
        response = jsonify({"error": "Rate limit exceeded", "retry_after": retry_after})
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response
    bucket.append(now)
    return None


def _prune_saved_runs():
    deleted = prune_old_runs(current_app.config["DATABASE_PATH"], current_app.config["MAX_SAVED_RUNS"])
    for run in deleted:
        if run.get("pdf_file_name"):
            cleanup_files(os.path.join(current_app.config["UPLOAD_FOLDER"], run["pdf_file_name"]))
        if run.get("preview_file_name"):
            cleanup_files(os.path.join(current_app.config["UPLOAD_FOLDER"], run["preview_file_name"]))
        if run.get("overlay_file_name"):
            cleanup_files(os.path.join(current_app.config["UPLOAD_FOLDER"], run["overlay_file_name"]))
    if deleted:
        logger.info("Pruned %s saved OCR runs", len(deleted))
