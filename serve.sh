#!/usr/bin/env bash
# Launch the NeuroZip live-inference Flask backend.
#
# Usage:
#   ./serve.sh                          # bind 0.0.0.0:8011 (LAN/Tailscale visible)
#   HOST=127.0.0.1 PORT=8000 ./serve.sh # local only
#
# Requires checkpoints to exist; run ./train.sh first.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8011}"

if [[ ! -d .venv ]]; then
  echo "no .venv yet — run ./train.sh first (or just create the venv)." >&2
  exit 1
fi
if [[ ! -f checkpoints/clip_proj.pt ]]; then
  echo "no projector checkpoint — run ./train.sh first." >&2
  exit 1
fi

echo "starting NeuroZip on http://${HOST}:${PORT}/"
exec $PY serve.py --host "$HOST" --port "$PORT"
