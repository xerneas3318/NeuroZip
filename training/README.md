# NeuroZip training (research code)

Self-contained training code for **both** NeuroZip codec families. This runs on a machine
with the THINGS-EEG dataset + a GPU (the desk box), not on a brew-installed laptop. The
trained checkpoints get dropped into `checkpoints/` (or `~/.neurozip/checkpoints`) and the
packaged `neurozip` app + the visualization load them.

## The two models

| Model | File | Bottleneck | Rate control |
|-------|------|-----------|--------------|
| **Continuous codec** (the "non" / original) | `codec.py` | scalar round() + factorized-Laplace entropy prior | learned, via `lambda_rate` |
| **RQ-VAE** (residual vector-quantized) | `rqvae.py` | n_q residual codebooks of K codes each | fixed: `n_q * 32 * log2(K)` bits |

Both share the same conv encoder/decoder and the same **frozen CLIP-space judge**, and both
are trained task-aware (the judge term routes gradient into the decoder).

## Train scripts

Continuous codec (fidelity baseline, then task-aware), as on the desk box:
```bash
python train.py projector                                   # the frozen judge (once)
python train.py codec  --lambda_task 0  --out fidelity_v4_high     # baseline
python train.py codec  --lambda_task 3 --init_from fidelity_v4_high --out neurozip_v4_high
```

RQ-VAE (warm-started from a trained continuous codec for speed):
```bash
python train_rqvae.py --epochs 6 --n_q 8 --codebook 512 \
    --lambda_task 1.0 --init_from fidelity_v4_high --out rqvae_high
```
- Rate is fixed by `--n_q` / `--codebook`. Example tiers:
  `--n_q 4 --codebook 256` (low) ... `--n_q 11 --codebook 512` (~matches continuous high bpp).
- Output: `checkpoints/rqvae_<tier>.pt` (+ a `.json` training log), with
  `config.model_type = "rqvae"` so the loader picks the right architecture.

## Loading from the packaged app
```python
from neurozip import runtime as rt
cont = rt.load_codec(tier="high", variant="neurozip")   # continuous
rq   = rt.load_rqvae("rqvae_high")                       # RQ-VAE
rt.available_rqvae()                                     # -> ['rqvae_high', ...]
```
