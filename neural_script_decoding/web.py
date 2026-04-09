import logging
import os
import json
from datetime import datetime

from flask import abort, current_app, flash, jsonify, redirect, render_template, request, send_from_directory, url_for

from .files import allowed_file, build_upload_paths, cleanup_files
from .services import diagnostics_snapshot, json_download_response, ollama_status, process_ocr_file
from .storage import delete_run, get_run, list_runs, prune_old_runs, save_run


logger = logging.getLogger(__name__)


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
    def api_ocr_async():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        paths = build_upload_paths(current_app.config["UPLOAD_FOLDER"], file.filename)
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

    @app.route("/api/ocr/jobs/<job_id>", methods=["GET"])
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
            "error": job["error"],
        }
        if job["result"]:
            result = job["result"]
            response.update(
                {
                    "run_id": result["run_id"],
                    "pdf_url": url_for("download_pdf", filename=result["pdf_file_name"]),
                    "preview_url": url_for("preview_image", filename=result["preview_file_name"]),
                    "overlay_url": url_for("overlay_image", filename=result["overlay_file_name"]) if result["overlay_file_name"] else "",
                    **result["payload"],
                }
            )
        return jsonify(response)

    @app.route("/health")
    def health_check():
        logger.info("Health check requested")
        return jsonify(
            {
                "status": "healthy",
                "initialized": _ocr_engine().initialized,
                "system": _system_status(),
                "background_jobs": _job_manager().summary(),
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
