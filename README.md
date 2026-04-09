# Neural Script Decoding

Neural Script Decoding is a Flask application for extracting text from handwritten images, comparing OCR engine output, correcting the selected text with a local LLM, and exporting the result as a PDF.

## Project Status

`app.py` is the canonical entrypoint. `app1.py` remains as a compatibility shim that exposes the same Flask app object.

## Features

- Multi-engine OCR orchestration across Tesseract, EasyOCR, and TrOCR
- Graceful fallback when optional OCR dependencies are not installed
- Local Ollama correction pass before PDF generation
- Optional background OCR job API for long-running requests
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
- Docker Engine plus Docker Compose if you want the containerized setup
- Enough disk space for the selected Ollama model

Practical disk guidance:

- `phi3:mini` is the default local model and is much smaller than `mistral`
- Ollama model downloads can consume multiple gigabytes
- On systems with a small `/var` partition, Docker and containerd storage may need to be moved to a larger filesystem such as `/home`

## Installation

Install the core web stack:

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install optional OCR models and test dependencies as needed:

macOS / Linux:

```bash
pip install -r requirements-ocr.txt
pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
pip install -r requirements-ocr.txt
pip install -r requirements-dev.txt
```

Initialize local configuration:

macOS / Linux:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
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

1. Install Tesseract OCR from a Windows distribution such as UB Mannheim.
2. Add the Tesseract install directory to `PATH`, for example `C:\Program Files\Tesseract-OCR`.
3. Open a new terminal and verify:

```powershell
tesseract --version
```

### Ollama Setup

macOS / Linux:

```bash
ollama serve
ollama pull phi3:mini
```

Windows PowerShell:

```powershell
ollama serve
ollama pull phi3:mini
```

To make this more repeatable, the repo now includes:

- `.env.example` for pinned local configuration
- `docker-compose.yml` for running the app and Ollama together
- `scripts/ollama_healthcheck.py` for simple readiness checks
- `scripts/setup_ollama.sh` for waiting on Ollama and pulling the configured model

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
| `BACKGROUND_OCR_WORKERS` | `2` | Number of in-process OCR background workers |
| `BACKGROUND_JOB_TIMEOUT` | `300` | Queue job timeout in seconds |
| `OCR_QUEUE_BACKEND` | `local` | `local` for in-process jobs, `redis` for external queue workers |
| `OCR_QUEUE_NAME` | `ocr` | Queue name used by async OCR jobs |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL for queued OCR jobs |
| `ENABLE_TESSERACT` | `true` | Enable the Tesseract engine |
| `ENABLE_EASYOCR` | `false` | Enable the EasyOCR engine |
| `ENABLE_TROCR` | `false` | Enable the TrOCR engine |
| `OLLAMA_STATUS_TIMEOUT` | `1.0` | Timeout for Ollama reachability checks |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `OLLAMA_MODEL` | `phi3:mini` | Ollama model name |

Example:

macOS / Linux:

```bash
export FLASK_SECRET_KEY="replace-me"
export OLLAMA_MODEL="phi3:mini"
export ENABLE_EASYOCR="true"
```

Windows PowerShell:

```powershell
$env:FLASK_SECRET_KEY="replace-me"
$env:OLLAMA_MODEL="phi3:mini"
$env:ENABLE_EASYOCR="true"
```

Recommended local setup:

1. Copy `.env.example` to `.env`.
2. Set `FLASK_SECRET_KEY` to a real secret.
3. Keep `OLLAMA_MODEL` pinned to the model you actually use.
4. For Docker Compose, keep `OLLAMA_URL` as-is in `.env`; Compose overrides it for the app container automatically.

## Running the Project

Minimal local development flow:

1. Create and activate the virtual environment.
2. Install `requirements.txt`.
3. Install Tesseract if you want OCR to work.
4. Start Ollama and pull `phi3:mini` if you want text correction.
5. Set any needed environment variables.
6. Start the app with `app.py`.
7. Open `http://127.0.0.1:5000`.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama serve
```

In a second terminal:

```bash
source .venv/bin/activate
ollama pull phi3:mini
export FLASK_SECRET_KEY="replace-me"
python3 app.py
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama serve
```

In a second PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
ollama pull phi3:mini
$env:FLASK_SECRET_KEY="replace-me"
python app.py
```

If you do not want Ollama correction yet, skip `ollama serve` and `ollama pull phi3:mini`. The OCR flow will still run, and the UI will show Ollama as unavailable.

Optional engines:

- Install `requirements-ocr.txt` and set `ENABLE_EASYOCR=true` to enable EasyOCR.
- Install `requirements-ocr.txt` and set `ENABLE_TROCR=true` to enable TrOCR.
- Leave both disabled if you only want the lighter Tesseract-based setup.

## Async OCR API

For heavier OCR or LLM correction requests, use the background job API instead of blocking on `/api/ocr`.

Create a job:

```bash
curl -F "file=@note.jpg" http://127.0.0.1:5000/api/ocr/jobs
```

The response returns `202 Accepted` with a `job_id` and `status_url`.

Poll job status:

```bash
curl http://127.0.0.1:5000/api/ocr/jobs/<job_id>
```

When the job completes, the status response includes the same OCR payload as `/api/ocr` plus `run_id`, `pdf_url`, `preview_url`, and `overlay_url`.

With `OCR_QUEUE_BACKEND=local`, this stays as an in-process queue for simple local development.

For a durable external queue, set `OCR_QUEUE_BACKEND=redis` and run a separate worker process. The included Docker Compose stack does that for you by starting Redis plus a dedicated OCR worker container.

## Docker Compose

Run the web app, Redis worker queue, and Ollama together:

```bash
cp .env.example .env
docker compose up --build -d
```

After Ollama starts, pull the configured model:

```bash
./scripts/setup_ollama.sh
```

Open:

```text
http://127.0.0.1:5000
```

Notes:

- The app container uses `http://ollama:11434/api/generate` internally.
- The Compose stack sets `OCR_QUEUE_BACKEND=redis` and `REDIS_URL=redis://redis:6379/0` for the app and worker.
- The `worker` service runs `scripts/run_ocr_worker.py` and processes queued OCR jobs outside the web process.
- Ollama data is stored in the named volume `ollama_data`.
- Uploaded files and the SQLite database are stored in named Docker volumes as well.
- If Docker commands fail with a permissions error, add your user to the `docker` group and start a new shell session.

### Storage Note

This project may require extra Docker storage when Ollama models are pulled.

On the reference Linux setup used during development, Docker and containerd storage were moved off `/var` and onto `/home`:

- Docker data root: `/home/satwik/docker-data/docker`
- containerd root: `/home/satwik/containerd-data/containerd`

If your `/var` partition is small and Ollama downloads fail with `no space left on device`, either:

1. switch to a smaller model such as `phi3:mini`
2. free space under `/var`
3. move Docker and containerd storage to a larger filesystem

To check Ollama readiness manually:

```bash
python3 scripts/ollama_healthcheck.py http://localhost:11434/api/tags
```

## Running Tests

macOS / Linux:

```bash
pytest
```

Windows PowerShell:

```powershell
pytest
```

The tests use mocked OCR and PDF generation, so they do not require model downloads or a running Ollama instance.
The suite also includes a real Tesseract integration test that generates a small fixture image at runtime and skips automatically when Tesseract is unavailable.

## Continuous Integration

GitHub Actions runs the test suite on Python 3.11 and 3.12 through [`.github/workflows/ci.yml`](/home/satwik/Neural-Script-Decodin/.github/workflows/ci.yml).

## Deployment

Linux production with Gunicorn:

```bash
gunicorn -c gunicorn.conf.py wsgi:application
```

Cross-platform production with Waitress:

macOS / Linux:

```bash
python3 serve.py
```

Windows PowerShell:

```powershell
python serve.py
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
- The application normalizes `UPLOAD_FOLDER`, `PDF_FONT_PATH`, and `DATABASE_PATH` to absolute paths so local runs behave consistently on Windows, macOS, and Linux.
