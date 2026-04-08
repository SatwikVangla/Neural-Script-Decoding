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
- Automated tests for upload, API, health, and download behavior
- GitHub Actions CI and a Gunicorn/Docker deployment path

## Code Layout

- `app.py`: stable public entrypoint
- `app1.py`: compatibility entrypoint exposing the same Flask app object
- `neural_script_decoding/app_factory.py`: Flask application construction
- `neural_script_decoding/web.py`: route handlers
- `neural_script_decoding/ocr.py`: OCR engine orchestration and dependency-aware initialization
- `neural_script_decoding/services.py`: LLM correction and PDF generation
- `neural_script_decoding/files.py`: upload path and cleanup helpers

## Requirements

- Python 3.11+
- Tesseract installed on the host if you want the Tesseract engine
- Ollama running locally if you want LLM correction
- EasyOCR and TrOCR are disabled by default and must be explicitly enabled

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

### Tesseract Setup

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

macOS with Homebrew:

```bash
brew install tesseract
```

Windows:

1. Install Tesseract OCR from a Windows distribution.
2. Add the Tesseract install directory to `PATH`.

### Ollama Setup

```bash
ollama serve
ollama pull mistral
```

## Configuration

The app reads configuration from environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLASK_SECRET_KEY` | random per process | Session signing |
| `UPLOAD_FOLDER` | `static/uploads` | Upload and generated PDF location |
| `MAX_CONTENT_LENGTH_MB` | `16` | Upload size limit in megabytes |
| `APP_HOST` | `0.0.0.0` | Host for the built-in Flask server |
| `APP_PORT` | `5000` | Port for the built-in Flask server |
| `APP_DEBUG` | `false` | Enable Flask debug mode |
| `LOG_LEVEL` | `INFO` | Application log level |
| `ENABLE_TESSERACT` | `true` | Enable the Tesseract engine |
| `ENABLE_EASYOCR` | `false` | Enable the EasyOCR engine |
| `ENABLE_TROCR` | `false` | Enable the TrOCR engine |
| `OLLAMA_STATUS_TIMEOUT` | `1.0` | Timeout for Ollama reachability checks |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `OLLAMA_MODEL` | `mistral` | Ollama model name |

Example:

```bash
export FLASK_SECRET_KEY="replace-me"
export OLLAMA_MODEL="mistral"
export ENABLE_EASYOCR="true"
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
The suite also includes a real Tesseract integration test that generates a small fixture image at runtime and skips automatically when Tesseract is unavailable.

## Continuous Integration

GitHub Actions runs the test suite on Python 3.11 and 3.12 through [`.github/workflows/ci.yml`](/home/satwik/Neural-Script-Decodin/.github/workflows/ci.yml).

## Deployment

Gunicorn entrypoint:

```bash
gunicorn -c gunicorn.conf.py wsgi:application
```

Docker build and run:

```bash
docker build -t neural-script-decoding .
docker run --rm -p 5000:5000 neural-script-decoding
```

The container installs Tesseract. Optional EasyOCR and TrOCR dependencies are still excluded by default; add `requirements-ocr.txt` to the image and set `ENABLE_EASYOCR=true` or `ENABLE_TROCR=true` if you want those engines enabled in deployment.

## Notes

- `static/uploads/` is intentionally gitignored except for `.gitkeep`.
- If only some OCR dependencies are installed, or if engines are disabled by configuration, the UI and `/health` endpoint will report that explicitly.
- `app.py` is the stable entrypoint for local runs and deployment wrappers.
- `app1.py` contains the actual application implementation.
