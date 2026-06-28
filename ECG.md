# ECG codec — the v4 architecture, a signal that *loves* reconstruction

Branch `ecg` (off `rian`). After EEG (compresses ~70×) and proteins (barely
compress — information-dense), ECG is the sweet spot: a **redundant, periodic,
continuous** `(leads × time)` signal. The same v4 codec reconstructs it almost
perfectly at high compression.

## Why ECG (and why not MRI)

This codec is a **1-D `(channels × length)`** compressor. It shines on signals
with redundancy/smooth structure and fails on dense or 2-D/3-D ones:

| modality | shape | redundancy | result |
|---|---|---|---|
| EEG | 63 × 250 | high (noise + smooth) | ~70× |
| **ECG** | **12 × 250** | **very high (heartbeat repeats)** | **31× @ 96% var** |
| protein seq | 20 × 250 | ~none (near-max-entropy) | ~1× |
| structural MRI | 2-D/3-D image | spatial, detailed, huge | wrong tool |

ECG is the *direct* analog of EEG: continuous amplitude, so the codec is rian's
`EEGCodec` with one change — **63 EEG channels → 12 ECG leads** — and the same
MSE + Laplace-rate training. `ecg_codec.py` reuses rian's exact blocks.

## Dataset

**PTB-XL** (PhysioNet), 12-lead clinical ECG at 100 Hz. `ecg_data.py` reads the
WFDB records and splits each 10 s recording into 250-sample (2.5 s) windows ->
`(12, 250)` epochs, per-lead z-scored.

## Result (held-out)

| metric | value |
|---|---|
| variance explained | **96%** |
| reconstruction MSE | **0.038** (normalized) |
| compression vs float16 | **31×** |

The reconstruction overlays the original almost exactly across all 12 leads — QRS
complexes, P and T waves preserved (`results/ecg_reconstruction.png`).

## Run

```bash
python ecg_data.py                                # build cache + summary
python train_ecg.py --epochs 80 --lambda-rate 0.02
python make_ecg_results.py                        # 12-lead reconstruction figure
```

## Takeaway

Three modalities, one architecture, no redesign — only the channel count changes
(63 EEG → 12 ECG → 20 amino acids). The v4 codec is a general 1-D
`(channels × length)` compressor, and how well it compresses is a direct readout
of how *redundant* the signal is: EEG and ECG compress hard, protein sequences
don't, and 2-D/3-D images (MRI) need a different (2-D) architecture entirely.
