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

## Result (held-out; spikes winsorized ±6σ, target band-limited to the BOLD band)

| metric | value |
|---|---|
| reconstruction MSE | **0.21** (raw target was 0.46) |
| variance explained | **56%** |
| compression vs float16 | **123×** |
| original / compressed | 25.6 KB → 0.21 KB per window |

## Lowering the MSE — the target was full of noise

The raw 0.46 MSE was the codec being forced to fit high-frequency noise. BOLD
lives below ~0.1 Hz, so band-limiting the target (a standard fMRI preprocessing
step) is the right move — and it cuts MSE ~4× (`results/fmri_smoothing.png`):

| temporal low-pass (σ, TR) | FWHM | MSE |
|---|---|---|
| 0 (raw) | 0 | 0.48 |
| 1.0 | ~5 s | 0.31 |
| **1.5** | **~7 s (≈ HRF width)** | **0.22** |
| 2.5 | ~12 s | 0.12 |

Honest caveat: filtering also shrinks the target's total variance, so
*variance-explained* rises only modestly (51%→56%) — the big win is in absolute
MSE, by not asking the codec to reconstruct noise. Other SNR levers that would
help further: a **coarser parcellation** (more voxels averaged per ROI),
**task fMRI** (event-locked, repeatable signal), or **attention across ROIs**
(predict each region from its functional-network neighbors).

## Why the MSE doesn't go lower — it's a noise floor, not a bit budget

The rate–distortion curve is **flat** (`results/fmri_rate_distortion.png`):

| capacity | MSE | variance explained | compression |
|---|---|---|---|
| c_lat=256 | 0.450 | 52% | 39× |
| **c_lat=128** | **0.459** | **51%** | **95×** |
| c_lat=64 | 0.558 | 40% | 317× |

Going 95× → 39× spends **2.4× more bits and improves MSE by 0.009**. The codec
already extracts essentially all the *compressible* structure; the remaining ~48%
is irreducible resting-state noise (thermal, physiological, residual motion). Two
diagnostics confirm it: motion spikes (|z|>5, just 0.35% of samples) carried 21%
of the squared energy before winsorizing, and the *median* window already
reconstructs at 0.33 MSE (67% variance) — the mean is dragged up by a few
motion-corrupted windows. So 95× is near rate–distortion-optimal: more bits are
wasted on noise. See the residual in `results/fmri_panels.png`.

## Where fMRI sits — the redundancy spectrum

How well the *same* v4 codec compresses is a direct readout of signal redundancy:

| modality | channels × length | variance explained | result |
|---|---|---|---|
| ECG | 12 × 250 | 96% | 31× (periodic) |
| EEG | 63 × 250 | 66% | 70× |
| **fMRI** | **200 × 64** | **51%** | **95×** (partial — noise floor) |
| protein seq | 20 × 250 | — | ~1× (max-entropy) |

## Run

```bash
python fmri_data.py                                # download ABIDE + build cache
python train_fmri.py --epochs 100 --c-lat 128 --hidden 192 --lambda-rate 0.01 --smooth 1.5
./serve_fmri.sh        # demo on http://localhost:8011/
```
