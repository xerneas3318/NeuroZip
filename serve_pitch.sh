#!/usr/bin/env bash
# Pitch entry point — the bio-hackathon hero page at `/`.
# Single screen: type a concept → see the brain recording fidelity loses
# and NeuroZip finds. Biology figures (ERP timeline + topographic head map)
# directly below. Drill-down links go to /dc#workspace and /dc#demo.
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
if [[ ! -f plots/phase1_erp_timeline.png ]]; then
  echo "missing plots/phase1_*.png — run:"
  echo "  $PY scripts/phase0_localization.py"
  echo "  $PY scripts/phase1_bio_figures.py"
  echo "to generate the biology figures before serving the pitch." >&2
  exit 1
fi

echo "==================================================="
echo "  NeuroZip — pitch (bio-hackathon hero)"
echo "  http://${HOST}:${PORT}/"
echo "  drill-downs: /dc  /clean  /dark"
echo "==================================================="
NEUROZIP_HOME=pitch exec $PY serve.py --host "$HOST" --port "$PORT"
