#!/usr/bin/env bash
# Train RQ-VAE + rian-v4 at several compression tiers for the viewer's tier slider.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

echo "=== RQ-VAE tiers ==="
[ -f checkpoints/codec_rqvae_200x.pt ] || $PY train_rqvae.py --num-quantizers 4  --epochs 70 --out checkpoints/codec_rqvae_200x.pt
[ -f checkpoints/codec_rqvae_100x.pt ] || $PY train_rqvae.py --num-quantizers 8  --epochs 70 --out checkpoints/codec_rqvae_100x.pt
# D=11 ~72x already exists as codec_rqvae_72x.pt

echo "=== v4 (rian) tiers ==="
[ -f checkpoints/codec_v4_200x.pt ] || $PY train_v4.py --lambda-rate 0.05  --epochs 15 --out checkpoints/codec_v4_200x.pt
[ -f checkpoints/codec_v4_100x.pt ] || $PY train_v4.py --lambda-rate 0.012 --epochs 15 --out checkpoints/codec_v4_100x.pt
# ~64x already exists as codec_v4_fidelity.pt

echo "=== tiers done ==="
for f in codec_rqvae_200x codec_rqvae_100x codec_rqvae_72x codec_v4_200x codec_v4_100x codec_v4_fidelity; do
  $PY -c "import torch;fv=torch.load('checkpoints/$f.pt',map_location='cpu',weights_only=False)['final_val'];print(f'$f: MSE={fv[\"mse\"]:.4f} ratio={fv[\"ratio\"]:.0f}x')" 2>/dev/null || true
done
