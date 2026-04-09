#!/usr/bin/env bash

set -euo pipefail

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-mistral}"
MAX_ATTEMPTS="${OLLAMA_MAX_ATTEMPTS:-30}"
SLEEP_SECONDS="${OLLAMA_RETRY_SECONDS:-2}"

echo "Waiting for Ollama at ${OLLAMA_BASE_URL}..."

attempt=1
until python3 scripts/ollama_healthcheck.py "${OLLAMA_BASE_URL}/api/tags"; do
  if [ "${attempt}" -ge "${MAX_ATTEMPTS}" ]; then
    echo "Ollama did not become ready after ${MAX_ATTEMPTS} attempts." >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep "${SLEEP_SECONDS}"
done

echo "Pulling Ollama model: ${OLLAMA_MODEL}"
OLLAMA_HOST="${OLLAMA_BASE_URL}" ollama pull "${OLLAMA_MODEL}"
echo "Ollama is ready with model ${OLLAMA_MODEL}"
