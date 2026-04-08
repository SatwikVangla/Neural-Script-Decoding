import logging
import os
from datetime import datetime

from flask import abort, current_app, flash, jsonify, redirect, render_template, request, send_from_directory, url_for

from .files import allowed_file, build_upload_paths, cleanup_files
from .services import build_result_payload, correct_text_with_llm, generate_pdf, json_download_response, ollama_status


logger = logging.getLogger(__name__)


def register_routes(app):
    @app.route("/")
    def index():
        return render_template("index.html", system_status=_system_status())

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
                ocr_results = _ocr_engine().process_with_all_engines(paths["file_path"])
                raw_text = _ocr_engine().combine_results(ocr_results)
                corrected_text = correct_text_with_llm(raw_text, current_app.config)
                generate_pdf(corrected_text, paths["pdf_path"], current_app.config["PDF_FONT_PATH"])
                cleanup_files(paths["file_path"], paths["preprocessed_path"])
                result_payload = build_result_payload(
                    filename=paths["filename"],
                    ocr_results=ocr_results,
                    raw_text=raw_text,
                    corrected_text=corrected_text,
                    system_status=_system_status(),
                )
                return render_template(
                    "results.html",
                    ocr_results=ocr_results,
                    raw_text=raw_text,
                    corrected_text=corrected_text,
                    pdf_path=paths["pdf_path"],
                    system_status=_system_status(),
                    result_payload=result_payload,
                )
            except Exception as exc:
                logger.error("Processing failed: %s", exc)
                flash(f"Error processing image: {exc}")
                cleanup_files(paths["file_path"], paths["preprocessed_path"])
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
            ocr_results = _ocr_engine().process_with_all_engines(paths["file_path"])
            combined_text = _ocr_engine().combine_results(ocr_results)
            corrected_text = correct_text_with_llm(combined_text, current_app.config)
            payload = build_result_payload(
                filename=paths["filename"],
                ocr_results=ocr_results,
                raw_text=combined_text,
                corrected_text=corrected_text,
                system_status=_system_status(),
            )
            return jsonify(
                {
                    "success": True,
                    **payload,
                }
            )
        except Exception as exc:
            logger.exception("API OCR failed")
            return jsonify({"error": str(exc)}), 500
        finally:
            if paths:
                cleanup_files(paths["file_path"], paths["preprocessed_path"])

    @app.route("/health")
    def health_check():
        logger.info("Health check requested")
        return jsonify(
            {
                "status": "healthy",
                "initialized": _ocr_engine().initialized,
                "system": _system_status(),
                "timestamp": datetime.now().isoformat(),
            }
        )

    @app.route("/download-json", methods=["POST"])
    def download_json():
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "No result payload provided"}), 400

        filename = payload.get("file", "ocr-result")
        stem = os.path.splitext(filename)[0] or "ocr-result"
        logger.info("Serving structured OCR export for %s", stem)
        return json_download_response(payload, f"{stem}.json")


def _ocr_engine():
    return current_app.extensions["ocr_engine"]


def _system_status():
    _ocr_engine().initialize_engines()
    return {
        "engines": _ocr_engine().available_summary(),
        "ollama": ollama_status(current_app.config),
    }
