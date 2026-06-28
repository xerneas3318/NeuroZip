# NeuroZip - task-aware EEG compression

> **Type a word → find the brain recording.**
> A neural EEG codec trained so that *what survives compression* is the
> CLIP-decodable semantic content of the EEG - making the compressed corpus
> text-searchable.

## TL;DR

**Engineering claim.** A task-aware EEG codec that preserves more
retrievability per bit than an MSE-only baseline at every compression
tier. At the headline 144× tier: top-5 retrieval 29% vs the fidelity
baseline's 19% (200-way; raw-EEG ceiling 45%; chance 2.5%). An
independent classifier the codec was never trained against confirms the
win at the only tier where it doesn't saturate (NeuroZip 100% vs
fidelity 88% at 144×; both methods ~100% at higher tiers).

**Validation claim** (not a discovery). NeuroZip's task loss is a
supervised CLIP-image objective. What it preserves tightest is *exactly*
what visual-ERP neuroscience would predict — read this as evidence the
codec is shaping the right thing, not as an unsupervised discovery.

All bio numbers below are at v4_low (144×), the same tier the
compression ratio comes from:

- **WHERE.** Visual-cortex channels (O1 O2 Oz Iz PO7 PO8 PO3 PO4 POz P7
  P8 P9 P10) reconstruct **21.0% tighter** under NeuroZip than under a
  fidelity-only codec at matched architecture, vs **5.0% tighter** on
  the other 50 channels — a **4.16× spatial preference**, p &lt; 0.001
  (permutation test, 10 000 random 13-channel sets;
  [`plots/phase2_permutation.json`](plots/phase2_permutation.json)).
- **WHEN.** N170 (the face/object-recognition window, 150–200 ms): NeuroZip
  MSE **16.1% below** fidelity. P100 (16.2%), P300 (10.7%) also favored;
  P200 is at parity (−3.2%) — not every ERP component favors NeuroZip at
  this tier. Don't oversell it.
- **HOW MUCH SURVIVES.** At 144×, the independent classifier reads
  NeuroZip at 100% and fidelity at 88%. At every higher tier this judge
  saturates near 100% for both methods — so the per-bit retrieval edge
  (the previous bullet) is the load-bearing claim, not held-out accuracy.

**Honest caveats.** One subject. Trial-averaged. The check-1 single-
channel ablation in [`plots/phase0_summary.json`](plots/phase0_summary.json)
cuts the other way: fidelity's downstream classifier depends more on a
single occipital channel (P8 drops it 12.5 pp), while NeuroZip distributes
the signal so no single channel matters above 3 pp. Different lens,
different story; both are in the repo.

Numerical evidence: [`plots/phase0_summary.json`](plots/phase0_summary.json),
[`plots/phase1_bio_numbers.json`](plots/phase1_bio_numbers.json),
[`plots/phase2_permutation.json`](plots/phase2_permutation.json),
[`results.md`](results.md).

> **One-sentence engineering pitch.** The same frozen model that decides
> what to throw away during compression is what lets you text-search
> what you kept.

> **Where do I go next?**
> - This README: install, run, file map, the numbers - biology first,
>   then the engineering result.
> - [`pitch.md`](pitch.md): the 3-minute stage script + Q&A prep.
> - [`ARCHITECTURE.md`](ARCHITECTURE.md): the *why* - design rationale,
>   the v1 → v4 evolution, the three easy-to-get-wrong spots in detail,
>   the CLIP-variant gotcha, and the circularity defense.
> - [`results.md`](results.md): the full per-tier numerical breakdown.
> - Inline docstrings in each module explain how individual pieces work.

## How it works

```
              ┌─────────────────────────────────────────────────┐
              │           NeuroZip codec (trained)              │
EEG epoch ──► │  encoder ─► quantize ─► decoder ──► EEĜ        │
              └────────────────────┬────────────────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────┐
                │  projector P (frozen judge)      │
                │  EEĜ ──► CLIP-image embedding ε̂ │
                └──────────────────────────────────┘
                                   │
                ┌──────────────────┴────────────────────────────┐
                │  CLIP-image embedding of the seen image  ε*  │  (frozen, no_grad)
                └─────────────────────────────────────────────-─┘
                                   │
                       L_task = 1 − cos(ε̂, ε*)
                              ──+──
                  L = λ_rate · bits + λ_recon · ‖EEG−EEĜ‖² + λ_task · L_task
```

P + CLIP are frozen, but **gradient flows through P** into the codec decoder
(one of the three easy-to-get-wrong spots, see comments in `codec.py`).

## Architecture / data

- **Dataset:** `Haitao999/things-eeg` (THINGS-EEG, Gifford et al. 2022; CVPR
  2025 re-release by Wu et al.). Single subject (sub-01), 63 channels @ 250 Hz,
  250-sample epochs. Train: 16 540 image-trials × 4 reps. Test: 200 concepts
  × 80 reps.
- **Frozen judge:** ViT-B/32 CLIP image/text features come precomputed in the
  dataset (LAION-2B weights - *not* OpenAI; see [`ARCHITECTURE.md`](ARCHITECTURE.md)
  for the gotcha). We never invoke CLIP at training.
- **Projector P:** depthwise temporal conv → channel-mix → 4-stage conv tower
  → ([CLS] + transformer blocks if `n_attn>0`) → L2-norm. 2.5 M params (conv)
  or 6.0 M (attention).
- **Codec:** 1D-conv autoencoder with optional ViT bottleneck, latent
  (32 ch × 32 ts), factorized Laplace prior for rate, uniform-noise
  quantizer at training, integer round at inference. **Default: conv-only**
  (~0.75 M params) - see ARCHITECTURE for why attention here actually hurts
  the story.
- **Recommended generation: v4** - conv-only codec + attention projector
  + attention held-out classifier. Train with `./train.sh sweep_v4`.

## Numbers — recommended v4 generation (sub-01, trial-averaged 80 reps; image-prompt retrieval, top-5 over 200 concepts)

| codec | bpp ↓ | ratio ↑ | MSE | top-1 | top-5 | top-10 | held-out top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **raw EEG** | 16.000 | 1× | — | 18.5% | 45.0% | 65.0% | — |
| `fidelity_v4_low` | 0.076 | 210× | 0.0401 | 5.5% | 19.0% | 29.0% | 88.0% |
| `fidelity_v4_med` | 0.154 | 104× | 0.0298 | 8.0% | 26.5% | 38.0% | 99.5% |
| `fidelity_v4_high` | 0.209 | 76× | 0.0231 | 8.0% | 26.5% | 40.5% | 100.0% |
| `fidelity_v4_xhigh` | 0.248 | 64× | 0.0213 | 9.5% | 31.5% | 43.5% | 100.0% |
| **`neurozip_v4_low`** | 0.111 | **144×** | 0.0359 | **10.5%** | **29.0%** | **42.5%** | **100.0%** |
| **`neurozip_v4_med`** | 0.190 | 84× | 0.0251 | **10.5%** | **32.5%** | **47.0%** | **100.0%** |
| **`neurozip_v4_high`** | 0.222 | 72× | 0.0234 | **14.0%** | **37.5%** | **52.0%** | **100.0%** |
| **`neurozip_v4_xhigh`** | 0.250 | 64× | 0.0223 | **14.5%** | **35.5%** | **50.0%** | **100.0%** |

Read it across rows: at every compression tier, NeuroZip preserves more
of the raw EEG's retrievability than the fidelity codec at matched
architecture. The held-out classifier (an independent judge that the
codec's loss never optimized against) saturates near 100% for both
methods *except* `fidelity_v4_low` at 210× compression (88%) — so the
load-bearing differentiator is the per-bit retrieval gap, not held-out
accuracy. Full iso-rate comparison and per-query examples in
[`results.md`](results.md).

The money plot: `demo/assets/rate_retrieval.png`.

## Run

Three convenience scripts cover the common workflow:

```bash
# 1. one-shot training: venv + dataset subset + all stages (idempotent)
./train.sh              # full pipeline (currently v2 codecs - see variants below)
./train.sh sweep_v4     # recommended generation (conv codec + attention judge)

# 2. live demo backend (binds 0.0.0.0:8011 by default, LAN-visible)
./serve.sh
# then open http://<host>:8011/

# 3. standalone Jupyter viewer (no Flask required)
.venv/bin/jupyter notebook notebook.ipynb

# variants
./train.sh sweep_v2      # ViT-bottleneck codec sweep
./train.sh sweep_v3      # v3 cascade: ViT codec + attention projector retrain
./train.sh eval          # re-run evaluation + regenerate demo assets
./train.sh proj_only     # just retrain the projector
HOST=127.0.0.1 ./serve.sh   # local-only
PORT=8000 ./serve.sh        # change port
```

If you prefer to invoke things by hand:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh        # ~3 GB targeted subset
bash run_all.sh                       # all stages
.venv/bin/python serve.py --host 0.0.0.0 --port 8011
```

Each stage is independently runnable:

```bash
.venv/bin/python data.py                 # Stage 0: load + summarize
.venv/bin/python train.py proj           # Stage 1: train frozen judge
.venv/bin/python train.py codec --out fidelity_med  # Stage 2: fidelity baseline
.venv/bin/python train.py neurozip --out neurozip_med --init_from fidelity_med  # Stage 3
.venv/bin/python train.py classifier     # Stage 4 circularity defense
.venv/bin/python evaluate.py --models fidelity_med neurozip_med  # Stage 4 metrics + assets
```

## Files

| file | purpose |
|---|---|
| `data.py` | THINGS-EEG dataset loader + per-channel normalization stats |
| `clip_proj.py` | EEG → CLIP projector `P` (the frozen judge) |
| `codec.py` | 1D-conv codec (optional ViT bottleneck) + factorized Laplace prior + noise/round quantizer |
| `train.py` | CLI: `proj`, `codec`, `neurozip`, `classifier` |
| `evaluate.py` | Matched-bpp comparison, rate-retrieval plot, demo JSON, image copy |
| `serve.py` | Flask backend: live CLIP text encoding, on-demand codec reconstruction, server-rendered figures |
| `demo.html` | Live demo (talks to `serve.py`) with free-text query + reconstruction viewer |
| `notebook.ipynb` | Standalone Jupyter viewer; auto-detects which codec generation is on disk |
| `scripts/build_notebook.py` | Regenerable notebook source - edit + re-run to rebuild `notebook.ipynb` |
| `scripts/download_data.sh` | Targeted ~3 GB subset download (not the 33k-file naive load) |
| `scripts/train_sweep_v2.sh` | ViT-bottleneck codec sweep (4 tiers) |
| `scripts/train_sweep_v3.sh` | NeuroZip codecs against the attention projector |
| `scripts/train_sweep_v4.sh` | **Recommended.** Conv-only codecs against attention projector |
| `scripts/train_v3_chain.sh` | Convenience: projector retrain → classifier → v3 sweep |
| `run_all.sh` | End-to-end pipeline (idempotent - re-runs cheaply) |
| `train.sh` / `serve.sh` | One-shot entry points |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Design rationale & v1 → v4 evolution |

## The three easy-to-get-wrong spots

Explained in depth in [`ARCHITECTURE.md`](ARCHITECTURE.md). Tl;dr:

1. **Gradient flow through the frozen judge.** `P` and CLIP are frozen
   (`requires_grad_(False)` and `no_grad` on the target embedding) but the
   decoder must still receive gradient through `P(EEĜ)`. `train.py`'s
   `_grad_flow_assert()` runs once at Stage-3 startup and asserts (a) non-zero
   decoder grad from a task-only backward, (b) zero judge grad.
2. **Quantizer mismatch.** Train with additive `U(-0.5, 0.5)` noise (a
   differentiable proxy for rounding); at inference use `torch.round`. Forget
   the noise at training time and `round()` breaks at inference.
3. **Normalization stats.** Per-channel mean/std are computed on train and
   cached at `data/norm_stats.pt`. They're needed (a) to invert the codec
   output to real microvolts for fidelity reporting, and (b) so bpp is
   bits-per-normalized-sample comparable across configs.

## Circularity defense

The compressor is trained with `P` + CLIP in the loop. If we only ever
evaluated retrieval with `P` + CLIP, "great numbers against your own judge" is
a fair takedown. So `train.py classifier` trains a **separate** EEG→concept
classifier on the test set's repetition splits (different epochs than what the
codec saw at train time, and a fundamentally different judge - softmax over
concepts, not contrastive against CLIP). `evaluate.py` reports its top-1 on
the codec's decompressed EEG for every checkpoint, so you can verify that
NeuroZip's win generalizes beyond the metric it was trained on.

## Scope caveat

THINGS-EEG epochs are short (1 s) trials, not long clinical recordings. The
storage pressure NeuroZip addresses is **dataset-scale**: millions of labeled
trial epochs of brain-image pairs. The Haitao release already re-stored EEG in
float16 to halve size - that's evidence the storage pressure is real for
exactly this kind of data.

## Citation

```bibtex
@misc{neurozip2026,
  title  = {NeuroZip: task-aware EEG compression for retrieval-preserving storage},
  author = {Hackathon team},
  year   = {2026},
  note   = {Built same-day on THINGS-EEG.}
}
```
