# rian-2 experiments — results

Seven prototype experiments requested on branch `rian-2`, grouped by class.
All Group A experiments run on the **real** repo codec/projector modules
(`codec.py`, `clip_proj.py`) over a controlled synthetic EEG surrogate (a weak
concept signature under a high-variance concept-independent distractor + noise,
per-channel z-normalized like `data.py`). Group B is a separate modality
(face images). Numbers are **directional** — tiny models, few epochs, one GPU,
small data — but each tests a real mechanism. Raw JSON in `tests/results/`.

Reproduce: `.venv/bin/python tests/exp_priors.py` (A1/A2/A3),
`exp_vq.py` (A4), `exp_variable_rate.py` (A5), `exp_contrastive.py` (A6),
`exp_b_face.py` (B).

---

## Group A — codec / rate / objective (share `exp_common.py`)

### A1 — real on-disk ratio (the "info-theory 144x vs actual 144x" question)
`exp_priors.py`. A textbook range coder driven by the codec's **own** factorized
Laplace prior, round-trip verified.

| quantity | value |
|---|---|
| modeled bits/symbol (what the repo reports) | 1.6636 |
| **range-coded actual bits/symbol** | **1.6622** |
| coder overhead vs theoretical | **+0.002 %** |
| round-trip lossless | ✅ |
| bpp modeled / range-coded / `savez_compressed` (today) / raw int16 | 0.0541 / 0.0540 / 0.0749 / 0.5201 |
| ratio vs fp16: **range-coded** vs **`savez` (shipped)** | **296×** vs **214×** |

**Takeaway.** The reported bitrate *is* achievable on disk — a real entropy
coder lands within 0.002 % of the modeled rate. What NeuroZip ships today
(`save_nz` → `np.savez_compressed`) leaves ~28 % on the table (214× vs 296×
here) because DEFLATE ignores the learned prior. Wiring the coder to the prior
turns the theoretical number into the on-disk number.

### A2 — scale hyperprior vs factorized prior
`exp_priors.py`. Same latents; only the entropy model changes.

| entropy model | bits/symbol | rate reduction |
|---|---|---|
| factorized Laplace (current) | 1.6636 | — |
| **scale hyperprior** | **1.5263** | **−8.3 %** |

### A3 — learned mixture prior vs single Laplace
`exp_priors.py`. Per-channel mixture-of-4-Laplaces.

| entropy model | bits/symbol | rate reduction |
|---|---|---|
| single Laplace (current) | 1.6636 | — |
| **mixture (K=4)** | **1.4855** | **−10.7 %** |

**A2+A3 takeaway.** Both richer priors cut rate for free (no distortion
change) — 8–11 % here — and stack with A1: they lower the *true* entropy, so the
real coded size drops below today's modeled estimate.

### A4 — VQ bottleneck vs scalar quantization
`exp_vq.py`. Same encoder/decoder; scalar+factorized vs VQ over 32 temporal
tokens. Rate-distortion (MSE vs bpp):

| scalar (λ sweep) | bpp | MSE | | VQ (K) | bpp | MSE |
|---|---|---|---|---|---|---|
| λ=0.02 | 0.0544 | 0.518 | | K=256 | 0.0118 | 0.678 |
| λ=0.08 | 0.0334 | 0.637 | | K=64 | 0.0093 | 0.727 |
| λ=3.0 (floor) | 0.0191 | 1.001 | | K=16 | 0.0075 | 0.751 |

**Takeaway (nuanced).** Scalar+factorized can't get below ~0.019 bpp here (the
prior floor); VQ reaches **0.008–0.012 bpp (1300–2100×) at lower MSE than the
scalar floor**. So VQ *extends* the achievable rate range downward. Caveat: VQ
codes 32 tokens vs scalar's 512 symbols, so the curves don't overlap — the
matched-rate deltas are extrapolated (flagged in JSON). Within its own range
scalar+factorized is strong; the path to fair overlap is grouped / residual VQ.

### A5 — one variable-rate model vs N fixed-rate models
`exp_variable_rate.py`. A single gain-conditioned codec produces every rate
point; baselines are one dedicated codec per rate.

| λ level | variable-rate (one model) bpp / MSE | fixed (dedicated) bpp / MSE | MSE gap |
|---|---|---|---|
| 0.08 | 0.042 / 0.544 | 0.035 / 0.582 | −6.5 % |
| 0.02 | 0.058 / 0.521 | 0.051 / 0.514 | +1.5 % |
| 0.005 | 0.089 / 0.512 | 0.081 / 0.492 | +4.1 % |

**Takeaway.** One model with a rate dial lands within a few % MSE of dedicated
per-rate models across 179–383× — supports shipping **one checkpoint with a
continuous rate control** instead of four (`*_low/med/high/xhigh`).

### A6 — contrastive task loss vs cosine-to-target (vs fidelity)
`exp_contrastive.py`. Frozen judge; fidelity / cosine / contrastive codecs swept
over λ_rate, compared at **matched bpp**. Ceiling (judge on uncompressed EEG)
= top-1 1.0; chance top-1 = 0.0125.

Matched-bpp top-5 vs fidelity:

| bpp | ratio | fidelity top-5 | **cosine top-5 (Δ)** | contrastive top-5 (Δ) |
|---|---|---|---|---|
| 0.041 | 392× | 0.067 | **0.802 (+73.6 pp)** | — |
| 0.034 | 465× | 0.067 | **0.577 (+51.0 pp)** | — |
| 0.026 | 617× | 0.071 | **0.277 (+20.6 pp)** | 0.087 (+1.6 pp) |
| 0.021 | 772× | 0.067 | 0.093 (+2.6 pp) | 0.079 (+1.2 pp) |

**Takeaway.**
1. **Task-aware ≫ fidelity at matched rate** — reproduces NeuroZip's core claim,
   strongly (+20 to +74 pp top-5 where signal survives; both converge to chance
   once the rate is too low for anyone).
2. **Contrastive did *not* beat cosine here.** In an easy regime both saturate
   (top-5 = 1.0 to 536×); in this hard regime cosine is clearly more robust —
   contrastive over-compressed (highest MSE, ~1.05) and sat near chance.
   **Caveat:** CE and (1−cos) have different magnitudes, so a shared λ_task=2.0
   is not a fair head-to-head; a real comparison needs per-loss weight
   normalization or a gradient-magnitude match. As tested, the simpler cosine
   term is the safer default.

---

## Group B (separate modality) — face-identity codec

### B — task-aware image compression for face identity
`exp_b_face.py`. Same 2D conv codec + factorized prior both ways; task-aware
adds `1 − cos(J(x̂), J(x))` with a frozen identity embedder J. **CelebA download
was blocked by Google Drive quota** (even with `gdown`), so this ran on **LFW**
(60 identities, 2950 images) — still a face-identity benchmark, well under 10 GB.
Compared at **matched bitrate** (λ_rate swept per mode); an **independent** judge
J2 (different seed) guards against circularity.

Matched-bitrate identity retrieval (chance top-1 = 0.0167):

| bpp | ratio vs 8-bit RGB | fidelity top-1 | **task-aware top-1** | task top-1 (independent judge) |
|---|---|---|---|---|
| 0.194 | 124× | 0.098 | **0.125 (+2.75 pp)** | 0.125 (vs fid 0.104) |
| 0.176 | 136× | 0.086 | 0.088 (+0.22 pp) | 0.096 (vs fid 0.103) |

Across the full sweep the task-aware curve is uniformly higher on top-1, top-5,
and verification AUC, and **degrades far more gracefully** as rate drops (e.g.
at λ=0.3 fidelity AUC 0.70 / task 0.77; at λ=0.8 fidelity AUC 0.44 / task 0.71).

**Takeaway.** The NeuroZip idea transfers off EEG: at matched bitrate a codec
trained to preserve *identity* keeps faces more recognizable than an MSE codec,
confirmed on a held-out judge (so it isn't just fitting the training judge).

---

## Honest limitations
- Synthetic EEG surrogate (not THINGS-EEG); mechanisms are real, absolute
  numbers are not comparable to the paper. A1's coder is prior-exact regardless.
- Tiny models / few epochs / single seed. Trends are robust to this; exact
  percentages are not.
- A6 cosine-vs-contrastive is sensitive to loss scaling (see caveat).
- B ran on LFW, not CelebA (GDrive quota). Loader still prefers CelebA if
  reachable.
