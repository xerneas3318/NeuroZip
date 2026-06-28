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

## Run

```bash
python train_rqvae.py --num-quantizers 11 --codebook-size 1024 --epochs 100 \
                      --out checkpoints/codec_rqvae_72x.pt        # D controls the tier
./serve_rqvae.sh        # viewer on http://localhost:8011/  (original vs RQ reconstruction)
```
