#!/usr/bin/env bash
# One-shot: set up venv, download data subset, run all training stages.
# Re-runs are idempotent — skips anything already cached.
#
# Usage:
#   ./train.sh             # full pipeline
#   ./train.sh sweep_v3    # just the v3 NeuroZip cascade (assumes v2 done)
#   ./train.sh proj_only   # just the projector retrain
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"

# ---- 1. virtualenv ----
if [[ ! -d .venv ]]; then
  echo "=== creating .venv (system-site-packages inherits torch) ==="
  python3 -m venv --system-site-packages .venv
  .venv/bin/pip install --quiet -r requirements.txt
fi

# ---- 2. data ----
if [[ ! -f data/Preprocessed_data_250Hz_whiten/sub-01/train.pt ]]; then
  echo "=== downloading dataset subset (~3 GB, one-time) ==="
  bash scripts/download_data.sh
fi

# ---- 3. dispatch ----
case "${1:-all}" in
  all)
    bash run_all.sh
    ;;
  sweep_v2)
    bash scripts/train_sweep_v2.sh
    ;;
  sweep_v3|v3)
    bash scripts/train_v3_chain.sh
    ;;
  sweep_v4|v4)
    # Conv-only codecs against the attention judge (cleanest NeuroZip story).
    if [[ ! -f checkpoints/clip_proj.pt ]]; then
      $PY train.py proj --epochs 40 --batch 512 --hidden 192 --lr 5e-4 \
          --n_attn 2 --attn_heads 4 2>&1 | tee logs/proj.log
    fi
    bash scripts/train_sweep_v4.sh
    if [[ ! -f checkpoints/holdout_classifier.pt ]]; then
      $PY train.py classifier --epochs 25 --steps_per_epoch 200 --batch 256 \
          --group 10 --n_attn 2 --attn_heads 4 2>&1 | tee logs/holdout.log
    fi
    $PY evaluate.py --models \
        fidelity_v4_low fidelity_v4_med fidelity_v4_high fidelity_v4_xhigh \
        neurozip_v4_low neurozip_v4_med neurozip_v4_high neurozip_v4_xhigh \
        2>&1 | tee logs/eval_v4.log
    ;;
  proj_only)
    $PY train.py proj --epochs 40 --batch 512 --hidden 192 --lr 5e-4 \
        --n_attn 2 --attn_heads 4 2>&1 | tee logs/proj.log
    ;;
  eval)
    $PY evaluate.py --models \
        fidelity_v2_low fidelity_v2_med fidelity_v2_high fidelity_v2_xhigh \
        neurozip_v3_low neurozip_v3_med neurozip_v3_high neurozip_v3_xhigh \
        2>&1 | tee logs/eval.log
    ;;
  *)
    echo "unknown subcommand: $1"
    echo "valid: all | sweep_v2 | sweep_v3 | sweep_v4 | proj_only | eval"
    exit 2
    ;;
esac
