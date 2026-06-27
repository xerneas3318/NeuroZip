#!/usr/bin/env bash
# NeuroZip end-to-end pipeline. Each stage saves a checkpoint and is runnable
# in isolation; re-running this script only retrains what isn't already cached
# on disk, then always regenerates evaluation + demo assets.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-.venv/bin/python}"

mkdir -p checkpoints logs demo/assets data

# ---- Stage 0: dataset (subset; only if not already downloaded) ----
if [[ ! -f data/Preprocessed_data_250Hz_whiten/sub-01/train.pt ]]; then
  bash scripts/download_data.sh
fi
$PY data.py

# ---- Stage 1: frozen judge (EEG -> CLIP projector P) ----
if [[ ! -f checkpoints/clip_proj.pt ]]; then
  $PY train.py proj --epochs 40 --batch 512 --hidden 192 --lr 5e-4 2>&1 | tee logs/proj.log
fi

# ---- Stages 2 + 3: fidelity + NeuroZip codecs (4 tiers, ViT bottleneck) ----
bash scripts/train_sweep_v2.sh

# ---- Stage 4 part 1: held-out classifier (circularity defense) ----
if [[ ! -f checkpoints/holdout_classifier.pt ]]; then
  $PY train.py classifier --epochs 25 --steps_per_epoch 200 --batch 256 --group 10 \
      2>&1 | tee logs/holdout.log
fi

# ---- Stage 4 part 2: evaluation + demo asset emission ----
$PY evaluate.py --models \
    fidelity_v2_low fidelity_v2_med fidelity_v2_high fidelity_v2_xhigh \
    neurozip_v2_low neurozip_v2_med neurozip_v2_high neurozip_v2_xhigh \
    2>&1 | tee logs/eval.log

echo
echo "Done. To view the demo, serve this directory:"
echo "  cd $(pwd) && python3 -m http.server 8000"
echo "  http://localhost:8000/demo.html"
