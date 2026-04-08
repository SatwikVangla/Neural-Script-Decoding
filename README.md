# Neural Script Decoding

Neural Script Decoding is a Flask application for extracting text from handwritten images, comparing OCR engine output, correcting the selected text with a local LLM, and exporting the result as a PDF.

## Project Status

`app.py` is the canonical entrypoint and delegates to `app1.py`, which contains the main application logic.

## Features

- Multi-engine OCR orchestration across Tesseract, EasyOCR, and TrOCR
- Graceful fallback when optional OCR dependencies are not installed
- Local Ollama correction pass before PDF generation
- Safer download handling and cleanup of temporary upload artifacts
- Responsive frontend for upload, review, and export flows
- Baseline automated tests for upload, health, and download behavior

## Requirements

- Python 3.11+
- Tesseract installed on the host if you want the Tesseract engine
- Ollama running locally if you want LLM correction

## Installation

Install the core web stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install optional OCR models and test dependencies as needed:

```bash
pip install -r requirements-ocr.txt
pip install -r requirements-dev.txt
```

## Configuration

The app reads configuration from environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLASK_SECRET_KEY` | random per process | Session signing |
| `UPLOAD_FOLDER` | `static/uploads` | Upload and generated PDF location |
| `MAX_CONTENT_LENGTH_MB` | `16` | Upload size limit in megabytes |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |

Example:

```bash
export FLASK_SECRET_KEY="replace-me"
export OLLAMA_MODEL="mistral"
```

## Running the App

Start the application:

```bash
python3 app.py
```

Then open `http://127.0.0.1:5000`.

## Running Tests

```bash
pytest
```

The tests use mocked OCR and PDF generation, so they do not require model downloads or a running Ollama instance.

## Notes

- `static/uploads/` is intentionally gitignored except for `.gitkeep`.
- If only some OCR dependencies are installed, the UI and `/health` endpoint will report which engines are currently available.
- `app.py` is the stable entrypoint for local runs and deployment wrappers.
- `app1.py` contains the actual application implementation.
