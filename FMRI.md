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

## Result (held-out)

| metric | value |
|---|---|
| variance explained | **51%** |
| compression vs float16 | **92×** |
| original / compressed | 25.6 KB → 0.3 KB per window |

The codec captures the **dominant network structure** (global signal, large-scale
networks) and compresses 92×, but resting-state BOLD carries a lot of
ROI-specific noise, so the reconstruction is partial — see the residual in
`results/fmri_panels.png` and the ROI time-courses in `results/fmri_traces.png`.

## Where fMRI sits — the redundancy spectrum

How well the *same* v4 codec compresses is a direct readout of signal redundancy:

| modality | channels × length | variance explained | result |
|---|---|---|---|
| ECG | 12 × 250 | 96% | 31× (periodic) |
| EEG | 63 × 250 | 66% | 70× |
| **fMRI** | **200 × 64** | **51%** | **92×** (partial) |
| protein seq | 20 × 250 | — | ~1× (max-entropy) |

## Run

```bash
python fmri_data.py                                # download ABIDE + build cache
python train_fmri.py --epochs 100 --c-lat 128 --hidden 192 --lambda-rate 0.008
./serve_fmri.sh        # demo on http://localhost:8011/
```
