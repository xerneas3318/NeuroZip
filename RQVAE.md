# RQ-VAE latent upgrade

Branch `rian-rqvae` (off `rian`). Upgrades the codec's **latent space** from
scalar-quantize + factorized-Laplace to **Residual Vector Quantization**
(SoundStream / EnCodec style).

## What changed

`rqvae.py`:
- `VectorQuantizerEMA` — a learned codebook with EMA updates (no codebook-loss tuning).
- `ResidualVQ` — `D` codebooks; each quantizes the residual of the previous ones.
- `EEGCodecRQ` — reuses rian's `EEGEncoder`/`EEGDecoder`; replaces the scalar
  quantizer + Laplace prior with `ResidualVQ`. Same reporting interface
  (`compress_then_reconstruct`, `latent_shape`, `bpp_floor`).

**Why it's an upgrade:** the latent dim `c_lat` no longer costs bits — the rate
is fixed by `D × log2(K)` per token (32 tokens), independent of `c_lat`. So you
decouple representational capacity from bitrate, and the discretization is
learned/data-adaptive instead of uniform rounding. Rate is set by `--num-quantizers`.

## Result (held-out, trial-averaged test MSE; README scale = single-trial-std norm)

| | latent | ratio | MSE | var-explained |
|---|---|---|---|---|
| README `neurozip_high` | scalar 32×32 + Laplace | 74× | 0.0246 | ~30% |
| **RQ-VAE** (`codec_rqvae_72x`) | **D=11 × K=1024 codes, 128×32 latent** | 72× | **0.0070** | **80%** |

3.5× lower MSE than the README at matched compression. Trained on the full
averaged train split (16,540 images) with scale augmentation.

## Benchmark over rian's v4

`train_v4.py` reproduces rian's v4 fidelity codec faithfully (conv-only EEGCodec,
`n_attn=0`, c_lat=32, **single-trial** training, 15 epochs) and evaluates it on
the same trial-averaged test MSE. All at matched ~65–72× compression:

| model | protocol | MSE | var-explained | ratio |
|---|---|---|---|---|
| **v4 (rian)** | single-trial, conv-only | 0.0259 | 27% | 64× |
| **RQ-VAE (ours)** | averaged + scale-aug, residual-VQ | **0.0064** | **80%** | 72× |
| fidelity (ours) | averaged + scale-aug, scalar | 0.0058 | 79% | 67× |

**RQ-VAE is ~4× lower MSE than v4** (`results/benchmark_vs_v4.png`). v4 reproduces
rian's README (their fidelity_high = 0.0266 @ 78×). The gap is mostly the training
protocol: v4 trains on single trials (magnitude ~1.0) but is scored on trial-
averaged EEG (magnitude ~0.19), so it over-shoots amplitude (27% variance, near
zero); our averaged + scale-augmented training fixes that magnitude gap. The
viewer shows all three side by side with per-image MSE.

## Run

```bash
python train_rqvae.py --num-quantizers 11 --codebook-size 1024 --epochs 100 \
                      --out checkpoints/codec_rqvae_72x.pt        # D controls the tier
./serve_rqvae.sh        # viewer on http://localhost:8011/  (original vs RQ reconstruction)
```
