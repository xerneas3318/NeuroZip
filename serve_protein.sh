#!/usr/bin/env bash
PY="${PY:-.venv/bin/python}"; PORT="${PORT:-8011}"
exec $PY serve_protein.py --host 0.0.0.0 --port "$PORT"
