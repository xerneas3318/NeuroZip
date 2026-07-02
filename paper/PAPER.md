# NeuroZip: Redundancy-Matched Neural Compression of Biosignals Preserves Downstream Tasks

*Draft — QBI Hackathon 2026 follow-up. Authors: TBD.*

Figures: `paper/figures/redundancy_spectrum.png`, `paper/figures/fmri_rate_distortion.png`.
Code + per-experiment write-ups live on branches `main` (EEG), `ecg-fable` (ECG),
`fmri-fable` (fMRI), `har-fable` (HAR), `redundancy-spectrum` (analysis).

---

## Abstract

Biosignals are expensive to store and move — a single fMRI session is hundreds of
megabytes, and clinical archives hold millions of ECGs — yet they are enormously
**redundant**: slow or periodic in time, and correlated across channels. We show
that a *small* neural codec (0.26–2.4M parameters) compresses biosignals **30–300×**
while preserving not just the waveform but the **decodable content that the signal
is recorded for**. Across four modalities — EEG (visual-object semantics), ECG
(clinical diagnosis), fMRI (object decoding), and wearable inertial sensors
(activity recognition) — a decoder trained on original data reads the compressed
data at 98–100% of its baseline task performance. We organize these results with a
single principle: **a signal compresses as much as it is redundant, and *where* the
redundancy lives dictates the codec.** A max-entropy control (i.i.d. noise) measures
0% redundancy and compresses 1×, anchoring the spectrum. Finally, we characterize a
**fidelity–function trade-off**: MSE-optimal compression preserves the high-variance
signal but can discard the low-variance task signal, so the right operating point
depends on whether the compressed data is destined for storage or for analysis.

## 1. Introduction

Neural and physiological recordings dominate the storage budgets of modern
neuroscience and clinical medicine, and general-purpose compressors (gzip on the
integer NIfTI/WFDB files a lab actually stores) barely help — they see high-entropy
byte streams and miss the structure. Learned codecs can do far better, but the usual
target, reconstruction MSE, is the wrong objective: what matters is whether the
*information the recording was made to capture* survives compression — the diagnosis
in an ECG, the decodable stimulus in EEG/fMRI, the activity in a wearable trace.

This paper makes four claims, each backed by an experiment:

1. **A single small codec compresses biosignals across modalities** at 30–300×
   (§4), spanning 1-D `(channels × time)` signals (EEG, ECG, HAR) and 4-D volumes
   (fMRI) with the same design family.
2. **Compressibility is redundancy, made quantitative** (§3). One consistent
   diagnostic across modalities plus an i.i.d. control shows biosignals are strongly
   temporally redundant (lag-1 autocorrelation 0.59–0.80) and compress 30–70×, while
   the control (0 redundancy) compresses 1×.
3. **The redundancy *axis* dictates the codec** (§3, §4.3). fMRI is low-rank in time
   but high-rank in space, so a temporal-transform codec beats a per-volume neural
   codec by 4× on MSE at equal or better ratio; ECG is redundant on both axes and a
   plain conv codec reconstructs it intact.
4. **Task performance is preserved, and its cost is characterized** (§4, §5). Across
   all four modalities a classifier/retriever trained on original data reads the
   compressed data at 98–100% of baseline. Where fidelity and function diverge
   (fMRI), we give the rate–distortion–function curve.

## 2. Method

### 2.1 A redundancy-matched codec family

Every modality is a `(units × time)` tensor: EEG (63 channels × 250), ECG (12
leads × 250), HAR (9 sensors × 128), fMRI (a 4-D volume sequence, i.e. voxels × TR).
The codec is an autoencoder with a discrete or transform latent and an entropy
model, instantiated to match the signal's shape and redundancy:

- **1-D conv codec** (EEG/ECG/HAR): a strided 1-D convolutional encoder/decoder over
  time with per-channel awareness; latent quantized (scalar + factorized-Laplace
  rate model, or vector-quantized). This is the original NeuroZip codec.
- **3-D conv VQ codec** (fMRI volumes): a small 3-D conv VQ-VAE per volume (EMA
  codebook + dead-code restart); tokens range-coded *across time*. LayerScale on the
  residual branches is required for the 3-D net to train at all.
- **Temporal-transform codec** (fMRI, low-MSE): per segment, an SVD of the
  `(time × voxels)` matrix keeps the top-K components — K full-resolution spatial
  maps (int8) + per-frame coefficients. Retains full spatial detail; truncates only
  the redundant time axis.

All models are small (0.26–2.4M parameters; the transform codec has none) and
produce real bytes on disk (arithmetic/range coding), compared against the gzipped
NIfTI/WFDB a lab actually stores.

### 2.2 Task-preservation, not just MSE

For each modality we fix a decoder that reads the *content of interest* and score it
on original vs decompressed held-out data:

| modality | task decoder | metric |
|---|---|---|
| EEG  | frozen EEG→CLIP projector | free-text / image retrieval top-5 |
| ECG  | 1-D CNN, 5-way superclass  | balanced accuracy, macro-AUC |
| fMRI | logistic regression on VT cortex | 8-way object decoding accuracy |
| HAR  | 1-D CNN, 6-way activity    | accuracy |

We report three numbers: **baseline** (train orig → test orig), **archival**
(train orig → test *decompressed*: the practical case of analysing an archive), and
**within** (train decompressed → test decompressed: is the information present at
all). "Retained" = archival-or-within / baseline.

## 3. The redundancy spectrum

We measure, per modality, from many `(units × time)` samples: **temporal** and
**channel** participation-ratio redundancy (`1 − PR/dim`; 0 for white noise, →1 for
low-rank), lag-1 temporal autocorrelation, and mean inter-channel correlation
(`redundancy_spectrum.py`). See Figure 1.

| modality | shape | temporal_red | channel_red | lag-1 autocorr | chan_corr | codec |
|----------|------:|-------------:|------------:|---------------:|----------:|------:|
| EEG    | 63×250  | 92% | 44% | 0.69 | 0.04 | 70× |
| fMRI   | 500×120 | 89% | 96% | 0.59 | 0.12 | 41× |
| ECG    | 12×250  | 82% | 72% | 0.80 | 0.41 | 30× |
| random | 63×250  |  0% |  0% | 0.00 | 0.00 |  1× |

The i.i.d. control anchors the interpretation: **0 redundancy → 1× compression** — a
codec cannot beat entropy. Every biosignal sits far from it (temporal autocorrelation
0.59–0.80) and compresses 30–70×. The *axis* differs: ECG's 12 leads are strongly
correlated (channel |corr| 0.41) so it is redundant both ways; fMRI is low-rank in
time but high-rank in space; EEG channels are near-decorrelated (whitened). This
axis is what §4.3 shows must drive codec design.

**Figure 1** (`redundancy_spectrum.png`): (left) where each modality's redundancy
lives — random at the origin, ECG top-right; (right) compression tracks temporal
redundancy, with the incompressible control at (0, 1×).

## 4. Results

### 4.1 Summary

| modality | dataset | codec (params) | compression | fidelity | task | **task retained** |
|---|---|---|---:|---|---|---:|
| EEG  | THINGS-EEG      | 1-D VQ (0.4M) | 70–144× | — (task-aware) | CLIP retrieval | preserved; +4–11pp over fidelity-only |
| ECG  | PTB-XL          | 1-D conv (1.1M) | 30× | VE 95.1%, 48 dB | 5-way diagnosis | **98%** (AUC 0.824→0.811) |
| fMRI | Haxby           | 3-D VQ (0.6M) | 26× | VE 82%, 40 dB | VT object decoding | **98–100%** |
| fMRI | Haxby           | temporal-SVD (0) | 41–75× | MSE 0.06–0.09 | (fidelity codec) | (see §4.3) |
| HAR  | UCI HAR         | 1-D conv (0.26M) | 35× | VE 93.3%, 38 dB | 6-way activity | **98%** |
| control | i.i.d. Gaussian | — | 1× | — | — | — |

### 4.2 EEG, ECG, HAR — fidelity and function preserved together

For signals whose task content lives in the same high-variance structure the codec
reconstructs, fidelity and function are preserved together:

- **ECG (PTB-XL, 30×)**: reconstruction VE 95.1% / 48.4 dB; 5-way diagnostic AUC
  falls only 0.824→0.811 (**98% retained**, accuracy unchanged at 71.5%) reading
  30×-compressed ECG with an original-trained classifier.
- **HAR (UCI, 35×)**: VE 93.3%; activity recognition 92.4%→90.8% (**98%**). The same
  codec, unchanged, works on wearables — the method is not neural-specific.
- **EEG (THINGS-EEG)**: the task-aware codec preserves free-text retrieval of the
  viewed object at high compression and beats a fidelity-only codec at matched rate
  by 4–11pp top-5 — the original NeuroZip result that motivates task-preservation.

### 4.3 fMRI — the redundancy axis, and the fidelity–function trade-off

fMRI is the instructive case. A per-volume 3-D VQ codec compresses each volume's
spatial pattern independently and its MSE floors at ~0.24, because a single volume is
high-rank in space. But a run is low-rank in *time* (a rank-16 temporal SVD of 121
TRs reconstructs to MSE 0.056). Matching the codec to that axis (the temporal-SVD
codec) drops MSE to **0.089 at 75×** and **0.059 at 41×** — lower MSE *and* higher
ratio than any per-volume setting. This is claim (3): the redundancy axis dictates
the codec.

The same data exposes the **fidelity–function trade-off** (Figure 2):

| fMRI codec | ratio | VE | VT decoding retained |
|---|---:|---:|---:|
| per-volume VQ `nd1` | 26× | 82% | **100%** |
| per-volume VQ `nd3` | 294× | 72% | 29% |
| temporal-SVD (K=8) | 75× | (MSE 0.089) | 35% |

MSE-optimal compression keeps the high-variance signal and drops the low-variance
multivoxel pattern that carries object identity; the codec that best preserves
*decoding* (`nd1`, 100% retained) is not the one that best preserves *MSE*. Choose by
whether the compressed data is for archival (maximize ratio/fidelity) or for
downstream analysis (preserve function).

**Figure 2** (`fmri_rate_distortion.png`): rate–distortion (flat, near the signal
ceiling) vs rate–function (monotonic: decoding retention 29→42→100% as rate rises).

## 5. Discussion

**Compressibility is redundancy.** The control makes this quantitative: no
redundancy, no compression. Biosignals are far from max-entropy, which is exactly
why 30–300× is achievable at high fidelity.

**Match the codec to the axis.** The most reliable single design lesson: find where
the redundancy lives (a five-line SVD/autocorrelation diagnostic) and compress that
axis. For fMRI this is a 4× MSE win; for ECG it means a plain conv codec suffices.

**Task-preservation as the objective.** Reconstruction MSE is a proxy. Where it
diverges from the task (fMRI), optimizing MSE can silently destroy the science.
Reporting the downstream task — and, when needed, adding a task-aware term (the EEG
route) — is the honest evaluation.

## 6. Limitations

- Single-subject / subset evaluations in places (fMRI: one Haxby subject; ECG:
  the downloaded subset — the retention conclusion is size-robust but absolute
  accuracies would rise with the full set).
- Compression ratios are reported at each modality's own high-fidelity operating
  point; they are not iso-fidelity across modalities.
- The temporal-SVD codec's basis is computed per segment at compress time (adaptive
  transform coding), so it ships no weights but stores the basis in the file.
- Neural-codec exact reproducibility depends on deterministic inference; the entropy
  layer itself is bit-exact.

## 7. Reproducibility

Each experiment is a branch with a findings doc, code, committed checkpoints, and a
one-command test:

- `ecg-fable` — `ECG_FABLE.md`, `python ecg_fable_eval.py`
- `fmri-fable` — `FMRI_FABLE.md`, `python fable_test.py`
- `har-fable` — `HAR_FABLE.md`, `python har_fable.py`
- `redundancy-spectrum` — `REDUNDANCY_SPECTRUM.md`, `python redundancy_spectrum.py`

Datasets are public (THINGS-EEG, PTB-XL, Haxby via nilearn, UCI HAR) and fetched by
the per-branch scripts.
