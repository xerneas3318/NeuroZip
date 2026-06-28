#!/usr/bin/env bash
# Launch the NeuroZip backend with the v2 (DC-design ported) frontend at /.
# Same Flask backend as serve.sh / serve_clean.sh, just routes / to
# NeuroZip.dc.html instead of demo.html / demo_clean.html. The v2 page
# includes a real codec workspace (compress / decompress an EEG epoch
# through any loaded codec) on top of the existing retrieval + viewer.
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

echo "==================================================="
echo "  NeuroZip — v2 design (IBM Plex, codec workspace)"
echo "  http://${HOST}:${PORT}/"
echo "  also: /clean, /dark, /dc (explicit aliases)"
echo "==================================================="
NEUROZIP_HOME=v2 exec $PY serve.py --host "$HOST" --port "$PORT"
