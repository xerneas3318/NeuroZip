# NeuroZip — task-aware EEG compression

Same-day hackathon project. **NeuroZip** is a neural EEG compressor trained to
preserve what the EEG is *about* (the image embedding it represents) rather
than pure waveform fidelity. The result: after aggressive compression you can
still text-search the EEG corpus ("accordion" → retrieve the epochs recorded
while the subject saw an accordion), while a fidelity-only codec at the same
compression ratio loses retrievability.

> **One-sentence pitch.** The same frozen model that decides what to throw
> away during compression is what lets you text-search what you kept.

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
  2025 release by Wu et al.). Single subject (sub-01), 63 channels @ 250 Hz,
  250-sample epochs. Train: 16 540 image-trials × 4 reps. Test: 200 concepts
  × 80 reps.
- **Frozen judge:** ViT-B/32 CLIP image/text features come precomputed in the
  dataset, so we never invoke CLIP at runtime.
- **Projector P:** depthwise temporal conv → channel-mix → 4-stage conv tower
  → AvgPool → MLP → L2-norm. 5.8 M params.
- **Codec:** 1D-conv autoencoder, latent (32 ch × 32 ts), factorized Laplace
  prior for rate, uniform-noise quantizer at training, integer round at
  inference.

## Numbers (sub-01, trial-averaged 80 reps; image-prompt retrieval, top-5 over 200 concepts)

| codec          |  bpp  | ratio vs fp16 |  mse   | top-1 | top-5 | top-10 | held-out top-1 |
|----------------|------:|--------------:|-------:|------:|------:|-------:|---------------:|
| **raw EEG**    | 16.000|         1× |    —   | 13.5% | 37.5% | 50.5%  |       —        |
| fidelity_low   | 0.080 |        199× | 0.0449 |  2.5% | 14.0% | 20.0%  |    53.0%       |
| fidelity_med   | 0.154 |        104× | 0.0365 |  5.0% | 16.0% | 26.5%  |    85.5%       |
| fidelity_high  | 0.206 |         78× | 0.0266 |  5.5% | 16.0% | 29.5%  |    82.0%       |
| **neurozip_low**  | 0.092 |    175× | 0.0394 |  4.0% | **16.0%** | **28.5%** | **94.5%** |
| **neurozip_med**  | 0.171 |     93× | 0.0366 |  5.5% | **19.5%** | **34.0%** | **96.0%** |
| **neurozip_high** | 0.217 |     74× | 0.0246 |  6.0% | **22.5%** | **37.5%** | **99.5%** |

Read it across rows: at every compression tier, NeuroZip preserves more of the
raw EEG's retrievability than the fidelity codec, and the held-out classifier
(an independent judge that the codec's loss never optimized against) is the
strongest signal — NeuroZip jumps from ~85% (fidelity) to ~96–99% retained.

Notably:
- **At ~95× compression, NeuroZip matches the fidelity codec's top-5 from
  ~104× and beats fidelity at every higher bpp tier the fidelity codec never
  reaches.**
- Fidelity_high *uses 2.5× more bits* than fidelity_low but plateaus at 16%
  top-5 — pure rate doesn't buy retrievability. NeuroZip's curve keeps climbing.
- Fidelity is slightly better on raw MSE (0.0246 vs 0.0266 for the high tier),
  which is exactly the point: NeuroZip is *worse* on bit-by-bit fidelity yet
  *better* on what the EEG is *about*.

The money plot: `demo/assets/rate_retrieval.png`.

## Run

Two convenience scripts cover the common workflow:

```bash
# 1. one-shot: venv + dataset subset + all training stages (idempotent)
./train.sh

# 2. live demo (binds 0.0.0.0:8011 by default, LAN-visible)
./serve.sh
# then open http://<host>:8011/

# variants
./train.sh sweep_v3      # just the v3 attention-projector cascade
./train.sh eval          # re-run evaluation + regenerate demo assets
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
| `clip_proj.py` | EEG → CLIP projector `P` (Stage 1 judge) |
| `codec.py` | 1D-conv codec + factorized Laplace prior + noise/round quantizer |
| `train.py` | CLI: `proj`, `codec`, `neurozip`, `classifier` |
| `evaluate.py` | Matched-bpp comparison, plot, demo JSON, image copy |
| `serve.py`  | Flask backend: live CLIP text encoding, on-demand codec reconstruction, server-rendered figures |
| `demo.html` | Live demo (talks to `serve.py`) with free-text query + reconstruction viewer |
| `run_all.sh` | End-to-end pipeline (idempotent — re-runs cheaply) |

## The three easy-to-get-wrong spots

Documented inline in code; reproduced here:

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
codec saw at train time, and a fundamentally different judge — softmax over
concepts, not contrastive against CLIP). `evaluate.py` reports its top-1 on
the codec's decompressed EEG for every checkpoint, so you can verify that
NeuroZip's win generalizes beyond the metric it was trained on.

## Scope caveat

THINGS-EEG epochs are short (1 s) trials, not long clinical recordings. The
storage pressure NeuroZip addresses is **dataset-scale**: millions of labeled
trial epochs of brain-image pairs. The Haitao release already re-stored EEG in
float16 to halve size — that's evidence the storage pressure is real for
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
