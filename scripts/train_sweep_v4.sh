#!/usr/bin/env bash
# v4 sweep: conv-only codecs trained against the attention projector.
#
# Rationale: with a ViT bottleneck inside the codec (v2), the *fidelity*
# baseline accidentally captures long-range temporal structure (~ERP windows)
# and steals some of NeuroZip's lead by encoding semantic content "for free"
# via architectural capacity. Reverting the codec to conv-only structurally
# protects the baseline, so NeuroZip's win is attributable to the actual
# contribution (task loss + frozen judge in the loop) and not to codec
# expressiveness.
#
# The PROJECTOR stays with attention (the judge gets stronger, gradients
# get sharper, retrieval improves at inference). That's where attention
# actually earns its keep without polluting the baseline.
#
# Prereq: checkpoints/clip_proj.pt must be the attention projector (built
# by run_all.sh / ./train.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
EPOCHS_FID=${EPOCHS_FID:-15}
EPOCHS_NZN=${EPOCHS_NZN:-15}
LAMBDA_TASK=${LAMBDA_TASK:-3.0}
mkdir -p logs

# Conv-only codecs converge ~2.5x faster than ViT codecs, so we can afford
# a bit more epochs without changing the total time budget.
TIERS=(
  "low    0.05    0.07"     # high compression (200x+)
  "med    0.01    0.015"
  "high   0.002   0.003"
  "xhigh  0.0003  0.0005"   # low compression (50x range)
)

for tier in "${TIERS[@]}"; do
  read -r tag fid_lr nzn_lr <<<"$tier"
  out_fid="fidelity_v4_${tag}"
  out_nzn="neurozip_v4_${tag}"
  echo
  echo "=== v4 tier $tag (conv-only codec | attention judge) ==="

  if [[ ! -f "checkpoints/${out_fid}.pt" ]]; then
    $PY train.py codec --epochs $EPOCHS_FID --batch 256 \
        --out $out_fid --lambda_rate $fid_lr --n_attn 0 --workers 4 \
        2>&1 | tee logs/${out_fid}.log
  else
    echo "  $out_fid exists; skipping"
  fi

  if [[ ! -f "checkpoints/${out_nzn}.pt" ]]; then
    $PY train.py neurozip --epochs $EPOCHS_NZN --batch 256 \
        --out $out_nzn --init_from $out_fid \
        --lambda_rate $nzn_lr --lambda_task $LAMBDA_TASK \
        --n_attn 0 --workers 4 \
        2>&1 | tee logs/${out_nzn}.log
  else
    echo "  $out_nzn exists; skipping"
  fi
done

echo
echo "=== v4 sweep done ==="
ls -lh checkpoints/*v4*.pt

echo
echo "Evaluate with:"
echo "  $PY evaluate.py --models \\"
echo "      fidelity_v4_low fidelity_v4_med fidelity_v4_high fidelity_v4_xhigh \\"
echo "      neurozip_v4_low neurozip_v4_med neurozip_v4_high neurozip_v4_xhigh"
