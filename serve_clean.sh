#!/usr/bin/env bash
# Launch the NeuroZip backend pointing at the clean / white-theme demo.
# Same Flask backend as serve.sh; just prints the /clean URL on startup.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8011}"

if [[ ! -d .venv ]]; then
  echo "no .venv — run ./train.sh first." >&2; exit 1
fi
if [[ ! -f checkpoints/clip_proj.pt ]]; then
  echo "no projector checkpoint — run ./train.sh first." >&2; exit 1
fi

echo "==============================================="
echo "  NeuroZip — clean (white-theme) interface"
echo "  http://${HOST}:${PORT}/"
echo "==============================================="
NEUROZIP_HOME=clean exec $PY serve.py --host "$HOST" --port "$PORT"
