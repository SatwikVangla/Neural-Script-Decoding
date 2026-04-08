import logging
import os
import textwrap

import requests

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover - dependency presence varies by environment
    FPDF = None


logger = logging.getLogger(__name__)


def correct_text_with_llm(text, config):
    if not text.strip():
        return ""

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
            return response.json()["response"].strip()
        return f"[Error: LLM response failed - Status {response.status_code}]"
    except requests.exceptions.RequestException as exc:
        logger.error("LLM request failed: %s", exc)
        return f"[Error: LLM offline - {exc}]"


def ollama_status(config):
    url = config.get("OLLAMA_URL")
    if not url:
        return {"configured": False, "reachable": False, "reason": "OLLAMA_URL is not configured"}

    try:
        response = requests.get(url.replace("/api/generate", "/api/tags"), timeout=config.get("OLLAMA_STATUS_TIMEOUT", 1.0))
        if response.ok:
            return {"configured": True, "reachable": True, "reason": "Ready"}
        return {"configured": True, "reachable": False, "reason": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as exc:
        logger.info("Ollama status check failed: %s", exc)
        return {"configured": True, "reachable": False, "reason": str(exc)}


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
