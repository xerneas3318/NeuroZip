#!/usr/bin/env bash
# Train a full rate sweep of fidelity + NeuroZip codecs at matched bpp tiers.
# NeuroZip uses a slightly higher lambda_rate at each tier to compensate for
# the task loss's mild bias toward keeping more bits.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
EPOCHS_CODEC=${EPOCHS_CODEC:-10}
EPOCHS_NZN=${EPOCHS_NZN:-10}
LAMBDA_TASK=${LAMBDA_TASK:-3.0}

mkdir -p logs

# (tier, fidelity_lambda_rate, neurozip_lambda_rate)
TIERS=(
  "low   0.05  0.07"   # high compression
  "med   0.01  0.015"  # medium
  "high  0.002 0.003"  # low compression
)
for tier in "${TIERS[@]}"; do
  read -r tag fid_lr nzn_lr <<<"$tier"
  echo "=== tier $tag (fid lr=$fid_lr, nzn lr=$nzn_lr) ==="
  $PY train.py codec    --epochs $EPOCHS_CODEC --batch 256 \
      --out fidelity_${tag} --lambda_rate $fid_lr --workers 4 \
      2>&1 | tee logs/fidelity_${tag}.log
  $PY train.py neurozip --epochs $EPOCHS_NZN   --batch 256 \
      --out neurozip_${tag} --init_from fidelity_${tag} \
      --lambda_rate $nzn_lr --lambda_task $LAMBDA_TASK --workers 4 \
      2>&1 | tee logs/neurozip_${tag}.log
done
echo "=== sweep done ==="
ls -la checkpoints/*.pt
