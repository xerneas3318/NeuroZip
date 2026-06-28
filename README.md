# NeuroZip

<div align="center">

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32%20LAION--2B-5A4FCF?style=flat-square)](https://github.com/mlfoundations/open_clip)
[![Flask](https://img.shields.io/badge/Demo-Flask-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Dataset: THINGS-EEG](https://img.shields.io/badge/Dataset-THINGS--EEG-003366?style=flat-square)](https://huggingface.co/datasets/Haitao999/things-eeg)
[![License: PolyForm NC](https://img.shields.io/badge/License-PolyForm%20Noncommercial-yellow.svg?style=flat-square)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

<img src="demo/assets/rate_retrieval.png" width="640" alt="Rate vs retrieval: NeuroZip preserves more retrievability per bit than a fidelity-only codec at every compression tier" />

**Task-aware EEG compression: keep the bits that mean something.**

</div>

## Table of Contents
- [System Overview](#system-overview)
- [The NeuroZip Method](#the-neurozip-method)
  - [Method Overview](#method-overview)
  - [The Task-Aware Loss](#the-task-aware-loss)
  - [Biological Localization](#biological-localization)
- [Results](#results)
  - [The Frozen Judge](#the-frozen-judge)
  - [Codec Results (v4)](#codec-results-v4)
  - [The Money Plot](#the-money-plot)
- [Repository Layout](#repository-layout)
- [Installation and Usage](#installation-and-usage)
  - [Quick Start](#quick-start)
  - [Training Pipeline](#training-pipeline)
  - [Live Demo](#live-demo)
- [Honest Caveats](#honest-caveats)
- [Acknowledgements](#acknowledgements)
- [Contributors](#contributors)

## System Overview

NeuroZip is a neural EEG compressor that throws away **waveform fidelity** to preserve **decodable meaning**: specifically, the CLIP embedding of the image the subject was looking at when the EEG was recorded. After roughly 144x compression you can still text-search the EEG corpus ("accordion" retrieves the epochs recorded while the subject saw an accordion), where a fidelity-only codec at the same compression ratio loses retrievability.

> **One sentence pitch.** The same frozen model that decides what to throw away during compression is what lets you text-search what you kept.

The end-to-end workflow:

1. **Train the frozen judge.** An EEG to CLIP-space projector, trained once and then frozen.
2. **Train the codec.** A 1D-conv autoencoder, with a task-aware loss that routes gradient through the frozen judge.
3. **Compress** EEG epochs into a compact integer latent (about 219 bytes per epoch at 144x).
4. **Search** the compressed corpus by free text or by image, through the same frozen judge.
5. **Reconstruct and visualize** any epoch and verify, with an independent classifier, that the meaning survived.

Why it matters: EEG datasets are exploding (millions of labeled brain-to-image trials). Storing them as raw float16 is wasteful, and existing lossy EEG codecs optimize MSE, which silently destroys the decodable content that is the whole reason the dataset exists. NeuroZip's loss function knows what the EEG is *for*.

## The NeuroZip Method

### Method Overview

The codec is a standard encoder, quantizer, decoder. The contribution is the **frozen judge** in the training loop: the reconstructed EEG is pushed through a frozen EEG-to-CLIP projector, and the codec is rewarded for landing near the CLIP embedding of the image the subject actually saw.

```
              +-------------------------------------------------+
              |            NeuroZip codec (trained)             |
EEG epoch --> |  encoder --> quantize --> decoder --> EEG_hat   |
              +------------------------+------------------------+
                                       |
                                       v
                  +----------------------------------------+
                  |  projector P (frozen judge)            |
                  |  EEG_hat --> CLIP-image embedding e_hat |
                  +----------------------------------------+
                                       |
                  +--------------------+-------------------------+
                  |  CLIP-image embedding of the seen image  e*  |  (frozen, no_grad)
                  +----------------------------------------------+
                                       |
                          L_task = 1 - cos(e_hat, e*)
```

P and CLIP are frozen, but **gradient flows through P into the codec decoder**. That gradient path is the single most important implementation detail (see [`ARCHITECTURE.md`](ARCHITECTURE.md)).

### The Task-Aware Loss

```
L = lambda_rate  * bits(latent | prior)
  + lambda_recon * || EEG - EEG_hat ||^2
  + lambda_task  * ( 1 - cos( P(EEG_hat), CLIP_image(seen) ) )
```

Term by term:

- **Rate term.** Bits estimated under a factorized Laplace prior with a learnable per-channel scale. Buys you compression.
- **Reconstruction term.** Standard MSE between input and reconstructed EEG. Anchors the codec so the output stays a real waveform.
- **Task term.** Cosine distance between the *reconstructed* EEG's CLIP embedding (via the frozen projector) and the *frozen* CLIP image embedding of what the subject saw. Buys you retrievability.

The fidelity baseline and the NeuroZip codec share the **same architecture and training budget**: only `lambda_task` differs (0 versus 3.0). That keeps the comparison clean: any difference is attributable to the task loss, not to model capacity.

### Biological Localization

Object identity in single-subject EEG is spatially and temporally localized, and NeuroZip's task loss concentrates preservation exactly where the visual system encodes object information.

- **WHERE.** Visual-cortex channels (O1, O2, Oz, Iz, PO7, PO8, and neighbors) reconstruct **32% tighter** under NeuroZip than under a fidelity-only codec at matched compression, versus only **7%** tighter on non-visual channels: a **4.7x spatial preference**.
- **WHEN.** Visual-evoked ERP windows reconstruct 12 to 25% tighter. The N170 (the face and object component, 150 to 200 ms) is **25.4% below** the fidelity baseline in MSE. P100, P200, and P300 are all also favored.
- **HOW MUCH SURVIVES.** A separate concept classifier, trained on a different data split and a different loss, reaches **100% top-1** on NeuroZip-decompressed EEG at 144x compression. The information needed to identify what someone looked at is low-dimensional and recoverable from a tiny fraction of the original signal.

Numerical evidence: [`plots/phase0_summary.json`](plots/phase0_summary.json) and [`plots/phase1_bio_numbers.json`](plots/phase1_bio_numbers.json).

## Results

All numbers are for subject `sub-01`, trial-averaged over 80 repetitions, on the recommended **v4** generation (conv-only codec, attention projector, attention held-out classifier). Chance on 200-way retrieval is 0.5% / 2.5% / 5.0% for top-1 / 5 / 10.

### The Frozen Judge

The projector is trained with symmetric InfoNCE against the dataset's precomputed CLIP image features, then frozen. Adding a [CLS] token plus a 2-block transformer head is the dominant design lever.

| projector | params | top-1 | top-5 | top-10 |
|---|---:|---:|---:|---:|
| conv-only (AvgPool + MLP head) | 2.5 M | 13.5% | 37.5% | 50.5% |
| **conv tower + [CLS] + 2 attn blocks** | **6.0 M** | **18.5%** | **45.0%** | **65.0%** |

The attention judge clears chance by 27x / 18x / 13x. Its scores (18.5 / 45.0 / 65.0) are the **no-compression ceiling**: every codec below should be read as a fraction of that.

### Codec Results (v4)

**Fidelity baseline (MSE + rate only):**

| codec | bpp | ratio | MSE | top-1 | top-5 | top-10 | held-out top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fidelity_v4_low` | 0.076 | 210x | 0.0401 | 5.5% | 19.0% | 29.0% | 88.0% |
| `fidelity_v4_med` | 0.154 | 104x | 0.0298 | 8.0% | 26.5% | 38.0% | 99.5% |
| `fidelity_v4_high` | 0.209 | 76x | 0.0231 | 8.0% | 26.5% | 40.5% | 100.0% |
| `fidelity_v4_xhigh` | 0.248 | 64x | 0.0213 | 9.5% | 31.5% | 43.5% | 100.0% |

**NeuroZip (MSE + rate + task):**

| codec | bpp | ratio | MSE | top-1 | top-5 | top-10 | held-out top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `neurozip_v4_low` | 0.111 | 144x | 0.0359 | **10.5%** | **29.0%** | **42.5%** | **100.0%** |
| `neurozip_v4_med` | 0.190 | 84x | 0.0251 | **10.5%** | **32.5%** | **47.0%** | **100.0%** |
| `neurozip_v4_high` | 0.222 | 72x | 0.0234 | **14.0%** | **37.5%** | **52.0%** | **100.0%** |
| `neurozip_v4_xhigh` | 0.250 | 64x | 0.0223 | **14.5%** | **35.5%** | **50.0%** | **100.0%** |

**The asymmetric trade (the whole thesis):** NeuroZip is worse on raw MSE by at most about 5%, and better on top-5 retrieval by +4 to +11 percentage points (roughly +13% to +42% relative). You pay a little waveform fidelity to keep a lot of meaning.

| tier | NZ bpp / fid bpp | ratio (NZ / fid) | top-5 lift |
|---|---|---|---|
| low | 0.111 / 0.076 | 144x / 210x | NZ +10.0 pp |
| med | 0.190 / 0.154 | 84x / 104x | NZ +6.0 pp |
| high | 0.222 / 0.209 | 72x / 76x | NZ +11.0 pp |
| xhigh | 0.250 / 0.248 | 64x / 64x | NZ +4.0 pp |

**In human terms.** A raw float16 epoch is 63 x 250 = 15,750 samples = 31.5 kB. A corpus of 1 million labeled trial epochs is about 30 GB at float16, and about **210 MB at 144x**, still retaining 29% top-5 retrievability and 100% held-out classifier accuracy.

### The Money Plot

[`demo/assets/rate_retrieval.png`](demo/assets/rate_retrieval.png): top-5 retrieval (y) versus bpp (x), both families. NeuroZip's curve sits cleanly above fidelity at every tier.

## Repository Layout

| file | purpose |
|---|---|
| `data.py` | THINGS-EEG loader and per-channel normalization stats |
| `clip_proj.py` | EEG to CLIP projector `P` (the frozen judge) |
| `codec.py` | 1D-conv codec (optional ViT bottleneck) + factorized Laplace prior + noise/round quantizer |
| `train.py` | CLI: `proj`, `codec`, `neurozip`, `classifier` |
| `evaluate.py` | Matched-bpp comparison, rate-retrieval plot, demo JSON, image copy |
| `serve.py` | Flask backend: live CLIP-text encoding, on-demand reconstruction, server-rendered figures |
| `demo.html`, `demo_clean.html` | Live demo pages (gov-website aesthetic) with free-text query and reconstruction viewer |
| `notebook.ipynb` | Standalone Jupyter viewer; auto-detects which codec generation is on disk |
| `scripts/train_sweep_v4.sh` | **Recommended** sweep: conv-only codecs against the attention projector |
| `scripts/download_data.sh` | Targeted dataset subset download |
| `train.sh`, `serve.sh`, `serve_clean.sh` | One-shot entry points |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Design rationale, the v1 to v4 evolution, the three gotchas, the CLIP-variant gotcha |
| [`results.md`](results.md) | Full results, biological localization, glossary |

## Installation and Usage

### Quick Start

```bash
git clone https://github.com/xerneas3318/NeuroZip.git
cd NeuroZip
git checkout criticism-1

python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
bash scripts/download_data.sh        # targeted THINGS-EEG subset (about 3 GB)
```

### Training Pipeline

```bash
./train.sh sweep_v4        # recommended generation, about 45 min on one RTX 4090
```

Each stage is independently runnable:

```bash
.venv/bin/python data.py                                            # Stage 0: load + summarize
.venv/bin/python train.py proj                                     # Stage 1: train frozen judge
.venv/bin/python train.py codec    --out fidelity_v4_med           # Stage 2: fidelity baseline
.venv/bin/python train.py neurozip --out neurozip_v4_med --init_from fidelity_v4_med  # Stage 3
.venv/bin/python train.py classifier                               # Stage 4: circularity defense
.venv/bin/python evaluate.py --models fidelity_v4_med neurozip_v4_med  # metrics + demo assets
```

`lambda_task = 3.0` for all NeuroZip codecs, `lambda_recon = 1.0` for all. The only per-tier dial is `lambda_rate` (see `scripts/train_sweep_v4.sh`).

### Live Demo

```bash
./serve_clean.sh           # binds 0.0.0.0:8011 by default
# then open http://<host>:8011/
```

The demo lets you type a free-text query, retrieve the matching EEG epochs through the frozen judge, reconstruct any epoch, and compare NeuroZip against the fidelity baseline tier by tier.

## Honest Caveats

The caveats are part of the story. They make the rest credible.

- **One subject.** `sub-01`. The pipeline is subject-agnostic (one flag in `data.py` switches subjects), but reported numbers are for one of the 10 subjects.
- **Trial-averaged eval.** Each of the 200 test concepts has 80 EEG repetitions; retrieval is computed on the average. Single-trial retrieval is much noisier. Trial-averaging is the standard THINGS-EEG protocol.
- **NeuroZip uses more bits at the low end** (0.111 versus 0.076 bpp). The fair framing is "NeuroZip at 144x beats fidelity at 210x by 10 points," not "matched bpp at the low tier." At the high and xhigh tiers the bpps are essentially matched.
- **The CLIP gotcha.** The dataset's "ViT-B-32" features are LAION-2B, not OpenAI (cosine 0.98 versus -0.06). The live-inference text encoder must use the matching variant or retrieval silently breaks. See [`ARCHITECTURE.md`](ARCHITECTURE.md).
- **Epochs are 1-second visual trials**, not long clinical recordings. Generalizing to continuous EEG would need a streaming codec (future work).

## Acknowledgements

- **THINGS-EEG** (Gifford, Dwivedi, Roig, Cichy 2022), as packaged on Hugging Face at [`Haitao999/things-eeg`](https://huggingface.co/datasets/Haitao999/things-eeg), with precomputed CLIP features.
- **OpenCLIP** and the **LAION-2B** ViT-B/32 weights (`laion2b_s34b_b79k`) used as the frozen semantic judge.
- Built at the **QBI Hackathon 2026** (Quantitative Biosciences Institute, UCSF).

## Contributors

NeuroZip was built at the QBI Hackathon 2026 by:

- Rian Butala
- Avinash Senthil
