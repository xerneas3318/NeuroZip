#!/usr/bin/env bash
# High-fidelity ViT sweep:
#   - 2 transformer blocks at the codec bottleneck (n_attn=2)
#   - 4 bpp tiers (added xhigh = very low compression)
#   - 30 epochs / tier on the fidelity baseline, 25 on the NeuroZip warm-start
# All outputs land in checkpoints/{fidelity,neurozip}_v2_{tier}.pt
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
EPOCHS_FID=${EPOCHS_FID:-30}
EPOCHS_NZN=${EPOCHS_NZN:-25}
LAMBDA_TASK=${LAMBDA_TASK:-3.0}
N_ATTN=${N_ATTN:-2}

mkdir -p logs

# (tier, fidelity_lambda_rate, neurozip_lambda_rate)
# Lower lambda_rate => more bits used => higher fidelity tier.
TIERS=(
  "low    0.05    0.07"     # high compression
  "med    0.01    0.015"
  "high   0.002   0.003"
  "xhigh  0.0003  0.0005"   # NEW: very-low compression, max fidelity
)
for tier in "${TIERS[@]}"; do
  read -r tag fid_lr nzn_lr <<<"$tier"
  out_fid="fidelity_v2_${tag}"
  out_nzn="neurozip_v2_${tag}"
  echo
  echo "=== tier $tag (n_attn=$N_ATTN, fid lr=$fid_lr -> $out_fid, nzn lr=$nzn_lr -> $out_nzn) ==="

  if [[ ! -f "checkpoints/${out_fid}.pt" ]]; then
    $PY train.py codec --epochs $EPOCHS_FID --batch 256 \
        --out $out_fid --lambda_rate $fid_lr --n_attn $N_ATTN --workers 4 \
        2>&1 | tee logs/${out_fid}.log
  else
    echo "  $out_fid exists; skipping"
  fi

  if [[ ! -f "checkpoints/${out_nzn}.pt" ]]; then
    $PY train.py neurozip --epochs $EPOCHS_NZN --batch 256 \
        --out $out_nzn --init_from $out_fid \
        --lambda_rate $nzn_lr --lambda_task $LAMBDA_TASK \
        --n_attn $N_ATTN --workers 4 \
        2>&1 | tee logs/${out_nzn}.log
  else
    echo "  $out_nzn exists; skipping"
  fi
done

echo
echo "=== v2 sweep done ==="
ls -lh checkpoints/*v2*.pt
