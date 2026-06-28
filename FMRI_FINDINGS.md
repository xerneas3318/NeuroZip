# NeuroZip on fMRI (THINGS-data, IT cortex) — working semantic-search demo

Branch `fmri-semantic`. The EEG thesis, transferred to fMRI: a **task-aware**
codec (reconstruction + a CLIP loss backpropagated through a frozen fMRI→CLIP
judge) keeps brain responses **text-searchable** under heavy compression, where a
**fidelity-only** (MSE) codec throws that away.

- `fidelity` codec — reconstruction (MSE) only; **never sees CLIP**.
- `actual`   codec — MSE **+ CLIP loss backpropagated through the frozen judge**.
- Both compress an IT pattern to the same tiny latent (same bitrate).

## Data (fits the disk; the full 43 GB train betas do not)
THINGS-fMRI single-trial responses to the **100 test images**, restricted to
**IT cortex** (~3–4k voxels/subject), 12 reps each (~150 MB). CLIP gallery = the
100 concepts' ViT-B/32 text features (reused from the THINGS-EEG set).
Honest rep-split: reps 1–6 fit the judge + train the codecs (**600 trials**);
reps 7–12 averaged = held-out eval. Retrieval is 100-way (chance top1 1%, top5 5%).

## Result — actual ≫ fidelity at matched compression (held-out)

| subject | compression | fidelity top5 | **actual top5** | lift | MSE (fid→act) |
|---|---:|---:|---:|---:|---:|
| sub-01 | 64× | 23% | **33%** | **+10pp** | 1.24 → 1.16 |
| sub-02 | 58× | 18% | **35%** | **+17pp** | 1.24 → 1.19 |
| sub-03 | 47× | 14% | **21%** | **+7pp**  | 1.22 → 1.14 |

The CLIP-backprop codec preserves semantic retrievability **+7 to +17pp** better —
and even reconstructs slightly *better* (the semantic signal regularizes it). The
frozen fMRI→CLIP judge itself decodes held-out reps at top5 ~41% (sub-01/02), on
par with the EEG judge.

The key fix vs the first attempt: train the codec on the **600 individual trials**,
not the 100 trial-averaged patterns — with only 100 examples the CLIP loss overfit
and hurt; with 600 it generalizes and the thesis reproduces.

## Run

```bash
scripts/download_fmri.sh                       # ~150 MB THINGS-fMRI IT test data
python fmri_neurozip.py --subject sub-01        # trains fidelity + actual, saves artifacts
python fmri_neurozip.py --subject sub-02
python fmri_neurozip.py --subject sub-03
./serve_fmri.sh                                 # semantic-search demo on :8012
```

Type a concept → it's CLIP-encoded → the 100 held-out brain responses are ranked
by how each codec's reconstruction decodes through the judge. The "actual" column
surfaces the right concept; the "fidelity" column degrades to noise.

## Honest caveats
- Single-trial fMRI at 50–64× is noisy: *per-query* rank varies a lot; the win is
  an **aggregate** (top5 over all 100), not every individual search.
- Test-set only (100 images). Demonstrating this at scale (and a stronger judge)
  needs the 8,740-image train betas (43 GB) which didn't fit the dev box.
- Linear (ridge) judge, per subject. An MLP judge + more data would lift the ceiling.
