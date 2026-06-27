#!/usr/bin/env bash
# v3 sweep: NeuroZip codecs trained against the new attention projector P_v2.
# Fidelity codecs (lambda_task=0) don't use the projector and are reused from
# the v2 sweep -- only NeuroZip variants need retraining.
#
# Prereqs:
#   - checkpoints/clip_proj.pt is the n_attn>0 projector (run train_projector_v2.sh)
#   - checkpoints/fidelity_v2_{low,med,high,xhigh}.pt exist
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
EPOCHS_NZN=${EPOCHS_NZN:-25}
LAMBDA_TASK=${LAMBDA_TASK:-3.0}
N_ATTN=${N_ATTN:-2}
mkdir -p logs

TIERS=(
  "low    0.07"
  "med    0.015"
  "high   0.003"
  "xhigh  0.0005"
)
for tier in "${TIERS[@]}"; do
  read -r tag nzn_lr <<<"$tier"
  init="fidelity_v2_${tag}"
  out="neurozip_v3_${tag}"
  if [[ ! -f "checkpoints/${init}.pt" ]]; then
    echo "  ERROR: missing $init -- run scripts/train_sweep_v2.sh first" >&2
    exit 1
  fi
  if [[ -f "checkpoints/${out}.pt" ]]; then
    echo "  $out exists; skipping"; continue
  fi
  echo
  echo "=== v3 tier $tag (init from $init, lr=$nzn_lr, lambda_task=$LAMBDA_TASK) ==="
  $PY train.py neurozip --epochs $EPOCHS_NZN --batch 256 \
      --out $out --init_from $init \
      --lambda_rate $nzn_lr --lambda_task $LAMBDA_TASK \
      --n_attn $N_ATTN --workers 4 \
      2>&1 | tee logs/${out}.log
done
echo
echo "=== v3 sweep done ==="
ls -lh checkpoints/neurozip_v3_*.pt
