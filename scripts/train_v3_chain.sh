#!/usr/bin/env bash
# After v2 codec sweep finishes, run the v3 cascade:
#   1. Back up v1 projector
#   2. Train new attention projector (n_attn=2)
#   3. Retrain held-out classifier with attention
#   4. Sweep v3 NeuroZip codecs (uses new projector as judge)
#   5. Re-evaluate all v2 + v3 models
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
mkdir -p logs

echo
echo "=== 1. backup v1 projector ==="
if [[ ! -f checkpoints/clip_proj_legacy.pt && -f checkpoints/clip_proj.pt ]]; then
  cp checkpoints/clip_proj.pt checkpoints/clip_proj_legacy.pt
  echo "  saved checkpoints/clip_proj_legacy.pt"
fi

echo
echo "=== 2. train attention projector (n_attn=2, hidden=192) ==="
$PY train.py proj --epochs 40 --batch 512 --hidden 192 --lr 5e-4 \
    --n_attn 2 --attn_heads 4 2>&1 | tee logs/proj_v2.log

echo
echo "=== 3. retrain held-out classifier (n_attn=2) ==="
rm -f checkpoints/holdout_classifier.pt
$PY train.py classifier --epochs 25 --steps_per_epoch 200 --batch 256 \
    --group 10 --n_attn 2 --attn_heads 4 2>&1 | tee logs/holdout_v2.log

echo
echo "=== 4. v3 NeuroZip sweep (warm-start from fidelity_v2, use new judge) ==="
bash scripts/train_sweep_v3.sh

echo
echo "=== 5. evaluation ==="
$PY evaluate.py --models \
    fidelity_v2_low fidelity_v2_med fidelity_v2_high fidelity_v2_xhigh \
    neurozip_v3_low neurozip_v3_med neurozip_v3_high neurozip_v3_xhigh \
    2>&1 | tee logs/eval_v3.log

echo
echo "=== v3 cascade done ==="
