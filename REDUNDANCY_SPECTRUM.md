# The redundancy spectrum — why (and where) biosignals compress

Branch `redundancy-spectrum`. This is the organizing analysis for the paper: a
signal compresses exactly as much as it is **redundant**, and *where* the
redundancy lives — across time vs across channels/space — dictates which codec
wins. We measure this with one consistent diagnostic across modalities plus a
max-entropy control.

## Diagnostic

For each modality we take many `(units × time)` samples and compute, scale-free
and dimension-normalized:

- **temporal redundancy** = `1 − participation_ratio(temporal covariance)/T`
- **channel redundancy**  = `1 − participation_ratio(channel covariance)/C`
- **lag-1 temporal autocorrelation** (temporal, comparable across modalities)
- **mean |inter-channel correlation|** (channel redundancy, comparable)

Participation ratio → 0 redundancy for white noise, → 1 for a low-rank signal.
`redundancy_spectrum.py`. Data: EEG (THINGS-EEG, 63×250), ECG (PTB-XL, 12×250),
fMRI (Haxby, voxels×TR), and i.i.d. Gaussian noise.

## Result

| modality | shape | temporal_red | channel_red | lag-1 autocorr | chan_corr | codec |
|----------|------:|-------------:|------------:|---------------:|----------:|------:|
| EEG    | 63×250  | 92% | 44% | 0.69 | 0.04 | 70× |
| fMRI   | 500×120 | 89% | 96% | 0.59 | 0.12 | 41× |
| ECG    | 12×250  | 82% | 72% | 0.80 | 0.41 | 30× |
| random | 63×250  |  0% |  0% | 0.00 | 0.00 |  1× |

(`plots/redundancy_spectrum.png` — left: where the redundancy lives; right:
compressibility vs temporal redundancy.)

Reading it:

- **The control is the anchor.** i.i.d. Gaussian noise measures 0% redundancy on
  every axis and compresses **1×** — a codec cannot beat entropy. Every real
  biosignal sits far from it and compresses **30–70×**. Compressibility is
  redundancy, made quantitative.
- **All biosignals are strongly temporally redundant** (lag-1 autocorrelation
  0.59–0.80; temporal participation-redundancy ≥ 82%) — they are slow/periodic
  relative to their sampling rate. That is the dominant, exploitable axis, and why
  temporal modelling matters (the `fmri-fable` result: compressing the *time* axis
  drove fMRI MSE from 0.24 → 0.06).
- **Channel redundancy separates the modalities.** ECG's 12 leads are strongly
  correlated (|corr| 0.41) — it is the most redundant signal and reconstructs
  almost losslessly. EEG channels are nearly decorrelated (0.04, it's whitened),
  fMRI voxels weakly correlated pairwise (0.12) but dominated by a few global
  spatial modes (channel participation-redundancy 96%).
- **Where the redundancy lives picks the codec.** fMRI: low-rank in time, high-rank
  in space → a temporal-transform codec wins over a per-volume one. ECG: redundant
  on both axes → a plain conv codec reconstructs it (and the diagnosis) intact.
  This is the paper's thesis in one figure.

## Run

```bash
pip install -r requirements.txt
# EEG (HF), ECG (PTB-XL), fMRI (Haxby) caches are produced by their branches;
# this script reads data_eeg/, data_ecg/, data_fmri/ and an i.i.d. control.
python redundancy_spectrum.py
```
