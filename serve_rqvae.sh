#!/usr/bin/env bash
# Serve the RQ-VAE reconstruction viewer on port 8011.
PY="${PY:-.venv/bin/python}"
PORT="${PORT:-8011}"
CKPT="${CKPT:-checkpoints/codec_rqvae_72x.pt}"
exec $PY serve_rqvae.py --host 0.0.0.0 --port "$PORT" --ckpt "$CKPT"
