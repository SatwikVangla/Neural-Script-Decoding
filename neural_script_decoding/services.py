import logging
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Response

from .files import persist_preview

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover - dependency presence varies by environment
    FPDF = None


logger = logging.getLogger(__name__)
PAYLOAD_VERSION = 2


def correct_text_with_llm(text, config):
    if not text.strip():
        return {
            "text": "",
            "used_llm": False,
            "status": "empty",
            "reason": "No OCR text available for correction",
        }

    prompt = f"""Please correct the following text for spelling and grammar errors.
This text was extracted from a handwritten document using OCR, so there may be character recognition errors.
Return only the corrected text without any explanations:

{text}"""

    try:
        response = requests.post(
            config["OLLAMA_URL"],
            json={"model": config["OLLAMA_MODEL"], "prompt": prompt, "stream": False},
            timeout=30,
        )
        if response.status_code == 200:
            return {
                "text": response.json()["response"].strip(),
                "used_llm": True,
                "status": "corrected",
                "reason": "Corrected with Ollama",
            }
        logger.warning("LLM response failed with status %s", response.status_code)
        return {
            "text": text,
            "used_llm": False,
            "status": "fallback",
            "reason": f"Ollama request failed with status {response.status_code}; using raw OCR text",
        }
    except requests.exceptions.RequestException as exc:
        logger.error("LLM request failed: %s", exc)
        return {
            "text": text,
            "used_llm": False,
            "status": "fallback",
            "reason": f"Ollama unavailable ({exc}); using raw OCR text",
        }


def ollama_status(config):
    url = config.get("OLLAMA_URL")
    if not url:
        return {"configured": False, "reachable": False, "reason": "OLLAMA_URL is not configured", "models": []}

    try:
        response = requests.get(url.replace("/api/generate", "/api/tags"), timeout=config.get("OLLAMA_STATUS_TIMEOUT", 1.0))
        if response.ok:
            payload = response.json()
            return {
                "configured": True,
                "reachable": True,
                "reason": "Ready",
                "models": [model.get("name") for model in payload.get("models", []) if model.get("name")],
            }
        return {"configured": True, "reachable": False, "reason": f"HTTP {response.status_code}", "models": []}
    except requests.exceptions.RequestException as exc:
        logger.info("Ollama status check failed: %s", exc)
        return {"configured": True, "reachable": False, "reason": str(exc), "models": []}


def generate_pdf(text, output_path, font_path):
    if FPDF is None:
        raise RuntimeError("FPDF is not installed")

    pdf = FPDF()
    pdf.add_page()
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("Arial", size=12)

    pdf.set_title("OCR Extracted Text")
    pdf.set_author("Multi-OCR App")
    pdf.set_creator("Handwriting Recognition System")

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
        for subline in textwrap.wrap(line, width=80):
            pdf.cell(0, 10, subline, new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)
    logger.info("PDF generated: %s", output_path)


def process_ocr_file(*, ocr_engine, config, paths, system_status):
    persist_preview(paths["file_path"], paths["preview_path"])
    overlay_data = ocr_engine.create_tesseract_overlay(paths["file_path"], paths["overlay_path"])
    ocr_results = ocr_engine.process_with_all_engines(paths["file_path"])
    raw_text = ocr_engine.combine_results(ocr_results)
    correction = correct_text_with_llm(raw_text, config)
    corrected_text = correction["text"]
    generate_pdf(corrected_text, paths["pdf_path"], config["PDF_FONT_PATH"])
    result_payload = build_result_payload(
        filename=paths["filename"],
        original_name=paths["original_name"],
        ocr_results=ocr_results,
        raw_text=raw_text,
        corrected_text=corrected_text,
        system_status=system_status,
        correction=correction,
        regions=overlay_data.get("regions", []),
    )
    return {
        "ocr_results": ocr_results,
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "correction": correction,
        "overlay_data": overlay_data,
        "result_payload": result_payload,
    }


def build_result_payload(
    *,
    filename,
    original_name,
    ocr_results,
    raw_text,
    corrected_text,
    system_status,
    correction=None,
    regions=None,
):
    selected_engine = None
    for result in ocr_results:
        if result.get("text") == raw_text and "error" not in result:
            selected_engine = result.get("engine")
            break

    return {
        "payload_version": PAYLOAD_VERSION,
        "file": filename,
        "original_file": original_name,
        "selected_engine": selected_engine,
        "ocr_results": ocr_results,
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "correction": correction or {},
        "regions": regions or [],
        "system": system_status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def normalize_result_payload(payload, *, created_at=None):
    normalized = dict(payload or {})
    normalized.setdefault("payload_version", 1)
    normalized.setdefault("ocr_results", [])
    normalized.setdefault("raw_text", "")
    normalized.setdefault("corrected_text", normalized.get("raw_text", ""))
    normalized.setdefault("regions", [])
    normalized.setdefault("system", {})
    normalized.setdefault("correction", {})
    normalized.setdefault("generated_at", created_at or datetime.now(timezone.utc).isoformat())
    return normalized


def diagnostics_snapshot(*, config, ocr_engine, system_status):
    upload_dir = Path(config["UPLOAD_FOLDER"])
    database_path = Path(config["DATABASE_PATH"])
    return {
        "app_env": config.get("APP_ENV", "development"),
        "upload_folder": {
            "path": str(upload_dir),
            "exists": upload_dir.exists(),
            "writable": os.access(upload_dir, os.W_OK) if upload_dir.exists() else False,
        },
        "database": {
            "path": str(database_path),
            "exists": database_path.exists(),
            "parent_writable": os.access(database_path.parent, os.W_OK),
        },
        "pdf_font": {
            "path": config["PDF_FONT_PATH"],
            "exists": os.path.exists(config["PDF_FONT_PATH"]),
        },
        "selected_ollama_model": config["OLLAMA_MODEL"],
        "available_ollama_models": system_status.get("ollama", {}).get("models", []),
        "engines_initialized": ocr_engine.initialized,
    }


def json_download_response(payload, download_name):
    import json

    return Response(
        json.dumps(payload, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )
