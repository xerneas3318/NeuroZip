# fMRI codec — the v4 architecture on brain ROI time series

Branch `fmri` (off `rian`). The raw fMRI **volume** is 3-D + time (huge, detailed
— the wrong shape for a 1-D codec). The right representation is the **parcellated
ROI time series**: `(regions × time)` — a `(channels × length)` signal exactly
like EEG. So the same v4 codec applies with one change: 63 EEG channels → **200
brain ROIs** (CC200 atlas). BOLD is continuous → MSE, same Laplace rate model.
`fmri_codec.py` reuses rian's exact blocks.

## Dataset

**ABIDE** preprocessed CC200 ROI time series (PhysioNet/FCP-INDI S3). `fmri_data.py`
downloads the per-subject `.1D` files (T TRs × 200 ROIs) and splits each into
overlapping 64-TR windows → `(200, 64)` epochs, per-ROI z-scored. ~800 subjects →
~3,900 windows.

## Result — constant compression (original-v4 latent budget), MSE driven down

Headline holds the compression at the **original v4 budget** (1024 latent symbols,
~83×) and gets the MSE down by improving the *architecture/data*, not the bitrate:

| metric | value |
|---|---|
| reconstruction MSE | **0.155** (started at 0.46 at this same compression) |
| variance explained | **68%** |
| compression vs float16 | **83×** (= original v4) |
| codec | `HeavyfMRICodec` — 14M params, residual + attention |

### What moved the MSE at *constant* compression (1024 symbols)

| change | MSE |
|---|---|
| original v4 (128×8, no denoise) | 0.46 |
| + band-limit target (smooth σ=1.5) | 0.21 |
| + smarter latent split (64×16, keeps temporal) | 0.169 |
| + full dataset (1034 subjects) | 0.160 |
| **+ heavier net (14M, residual+attention)** | **0.155** |

The heavier architecture helps only **marginally** (0.160 → 0.155) — and a 33M
version *overfits* (0.212). So at this compression the model is **data/rate-limited,
not capacity-limited**: a conv v4 is already near-optimal for the bitrate, and the
real wins were denoising the target + a smarter latent split + more data. To go
materially lower you must spend bits (`fmri_codec_v5.py` / big-c_lat reach ~0.10 at
8×) or change the signal (coarser atlas / task fMRI).

## Why the MSE started high — and what actually fixed it

I diagnosed it instead of guessing (PCA on the held-out data vs the codec):

- **The codec was already ~linear-optimal for its bitrate.** At matched
  compression, linear PCA gives MSE ~0.23 and the codec gave 0.21 — so it wasn't
  underfitting. The data is just **high-rank**: spatial PCA needs 20 components
  for 56% variance, 100 for 87%. Resting-state ROIs are largely independent, so
  there isn't much redundancy to exploit — that's the honest reason it's harder
  than ECG/EEG.
- **But two things were inflating MSE and were fixable** (`results/fmri_mse_improvement.png`):
  1. **Noise in the target.** BOLD is <0.1 Hz; high-freq content is noise the
     codec was forced to fit. Band-limiting (σ=1.5 ≈ HRF width) → 0.46 → 0.21.
  2. **An 8× temporal bottleneck.** 64 TRs → 8 latent timepoints kept only 70% of
     the temporal variance *before the codec even tried.* Keeping more (4×→16,
     2×→32 latent timepoints) improves **both** MSE and variance-explained:

     | temporal | latent_t | MSE | variance | ratio |
     |---|---|---|---|---|
     | 8× | 8 | 0.204 | 57% | 120× |
     | **4×** | 16 | **0.163** | **65%** | 58× |
     | 2× | 32 | 0.140 | 70% | 29× |

Net: **MSE 0.50 → 0.163, variance 51% → 65%.** The remaining error is the genuine
high-rank floor; pushing lower means spending bits (2× temporal → 0.140 @ 29×) or
changing the signal — coarser parcellation, task fMRI (repeatable), or ROI
attention.

## Aside: spending more *spatial* bits is flat — temporal was the lever

Before finding the temporal bottleneck, sweeping spatial capacity (`c_lat`, at a
fixed 8× temporal) looked like a noise floor (`results/fmri_rate_distortion.png`):

| c_lat (8× temporal) | MSE | variance | compression |
|---|---|---|---|
| 256 | 0.450 | 52% | 39× |
| 128 | 0.459 | 51% | 95× |
| 64 | 0.558 | 40% | 317× |

39× → 95× barely moved MSE — so *more spatial channels* don't help. That's what
made the **temporal** axis the real fix (above): the 8× downsampling, not the
spatial latent, was the binding constraint. Also worth noting: the *median*
window already reconstructs at 0.33 MSE — the mean is dragged up by a few
motion-corrupted windows.

## Where fMRI sits — the redundancy spectrum

How well the *same* v4 codec compresses is a direct readout of signal redundancy:

| modality | channels × length | variance explained | result |
|---|---|---|---|
| ECG | 12 × 250 | 96% | 31× (periodic) |
| EEG | 63 × 250 | 66% | 70× |
| **fMRI** | **200 × 64** | **68%** | **83×** (constant; heavier net) |
| protein seq | 20 × 250 | — | ~1× (max-entropy) |

## Run

```bash
python fmri_data.py                                # download ABIDE + build cache
python train_fmri.py --epochs 160 --arch heavy --c-lat 64 --hidden 160 --depth 2 --n-attn 2 --n-down 2 --lambda-rate 0.005 --smooth 1.5 --wd 1e-4
./serve_fmri.sh        # demo on http://localhost:8011/
```
