#!/usr/bin/env bash
# NeuroZip end-to-end. Idempotent — skips anything already on disk.
# Order matters: projector first (used as judge by Stage-3 NeuroZip codecs).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"
mkdir -p checkpoints logs demo/assets data

# ---- Stage 0: dataset ----
if [[ ! -f data/Preprocessed_data_250Hz_whiten/sub-01/train.pt ]]; then
  bash scripts/download_data.sh
fi
$PY data.py

# ---- Stage 1: EEG -> CLIP projector with [CLS] + attention head ----
if [[ ! -f checkpoints/clip_proj.pt ]]; then
  $PY train.py proj --epochs 40 --batch 512 --hidden 192 --lr 5e-4 \
      --n_attn 2 --attn_heads 4 2>&1 | tee logs/proj.log
fi

# ---- Stages 2+3: fidelity + NeuroZip codecs (4 tiers, ViT bottleneck) ----
bash scripts/train_sweep_v2.sh

# ---- Stage 4 part 1: held-out classifier (circularity defense) ----
if [[ ! -f checkpoints/holdout_classifier.pt ]]; then
  $PY train.py classifier --epochs 25 --steps_per_epoch 200 --batch 256 \
      --group 10 --n_attn 2 --attn_heads 4 2>&1 | tee logs/holdout.log
fi

# ---- Stage 4 part 2: evaluation + demo asset emission ----
$PY evaluate.py --models \
    fidelity_v2_low fidelity_v2_med fidelity_v2_high fidelity_v2_xhigh \
    neurozip_v2_low neurozip_v2_med neurozip_v2_high neurozip_v2_xhigh \
    2>&1 | tee logs/eval.log

echo
echo "Done. Start the live demo:"
echo "  ./serve.sh"
