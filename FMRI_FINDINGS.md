# NeuroZip on fMRI (THINGS-data, IT cortex) — findings

Branch `fmri-semantic`. Goal: transfer the EEG fidelity-vs-task-aware semantic
demo to fMRI, on a dataset where subjects view object images (THINGS-fMRI),
mirroring the THINGS-EEG pipeline.

## Setup (what fits on disk)
- THINGS-fMRI single-trial responses to the **100 test images**, restricted to
  **IT cortex** (~3–4k voxels/subject), 12 reps each. 149 MB (the full 8,740-
  image train betas are 43 GB and do not fit in 25 GB free disk).
- CLIP gallery: the 100 concepts' ViT-B/32 text features (100/100 matched to the
  THINGS-EEG feature set, reused).
- Honest rep-split: reps 1–6 fit the judge + train the codec; reps 7–12 averaged
  = held-out eval. Retrieval is 100-way (chance: top1 1%, top5 5%).

## What transfers (works)
- **fMRI signal is strong:** split-half image-identity retrieval = top1 9–14% /
  top5 21–37% — IT reliably encodes which image was seen.
- **The judge transfers:** a frozen ridge fMRI→CLIP map, fit on reps 1–6 and
  evaluated on reps 7–12, retrieves at **top1 ~15% / top5 ~41–45%** (sub-01/02)
  — on par with the EEG judge (top5 45%).

## What does NOT transfer (the thesis, on this data)
At matched compression, the task-aware codec does **not** beat fidelity:

| sub-02 | compression | fidelity top5 | task-aware top5 | lift |
|---|---:|---:|---:|---:|
| latent 64  | 58× | 34.0 | 36.0 | +2.0 |
| latent 128 | 29× | 39.0 | 34–37 | −2 to −5 |
| latent 256 | 15× | 43.0 | 33.0 | −10.0 |

Two reasons, both from the 100-image / test-only constraint:
1. **No headroom** — at 15× the *fidelity* codec (top5 43%) already matches the
   judge ceiling (41%); it's near-lossless for retrieval, so task-aware can't add.
2. **Task loss overfits** — trained on 100 images through a linear judge, the CLIP
   term distorts the codec and *hurts* held-out retrieval. MSE also floors at
   ~0.92 (var-exp ~8%): IT single-trial patterns are noise-dominated.

## Bottom line
The **signal and the judge** transfer to fMRI cleanly. Demonstrating the
**fidelity-vs-task-aware advantage** needs a *training* set to fit a codec whose
task-aware variant generalizes — i.e. the full THINGS-fMRI train betas (disk-
blocked here) or a dataset that ships a train/test split small enough to fit
(e.g. Kamitani Generic Object Decoding). `fmri_neurozip.py` is the working,
reusable pipeline; drop in train data and the comparison becomes meaningful.
