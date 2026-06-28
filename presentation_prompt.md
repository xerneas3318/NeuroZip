# NeuroZip — Presentation Brief for Claude

> **Your job, Claude:** design a presentation for NeuroZip. This document is
> your single source of truth — every fact, number, claim, narrative beat, and
> visual reference you need is below. Build a slide deck (or a single
> long-form web presentation, depending on how the user asks) that lands the
> story crisply for a technical-but-not-specialist audience.

---

## 0. Quick orientation

- **What it is.** NeuroZip is a neural EEG compressor that throws away
  *waveform fidelity* to preserve *semantic meaning* — specifically, the CLIP
  embedding of the image the subject was looking at when the EEG was recorded.
  After ~144× compression you can still text-search the EEG corpus
  ("accordion" → retrieves the epochs recorded while the subject saw an
  accordion).
- **Why it matters.** EEG datasets are exploding (millions of labeled
  brain↔image trials). Storing them as raw fp16 is wasteful. Existing lossy
  EEG codecs optimize MSE, which silently destroys the decodable content
  *that's the whole reason the dataset exists*. NeuroZip's loss function knows
  what the EEG is *for*.
- **Built in:** one hackathon day, single subject (`sub-01`), single RTX 4090.
- **The pitch in one sentence.** *The same frozen model that decides what to
  throw away during compression is what lets you text-search what you kept.*

---

## 1. The audience & tone

**Audience.** Smart engineers / ML researchers who don't necessarily know EEG.
Assume they know what CLIP is, what an autoencoder is, what rate-distortion
means. Don't assume they know what an ERP is or what THINGS-EEG is — define
once when first used.

**Tone.** Confident, specific, slightly dry. Numbers do the work; adjectives
don't. Each slide should land one claim with the evidence on the same slide.
No "in this presentation we will…" preambles. No bullet soup.

**What to avoid.**
- Don't oversell. The honest caveats (one subject, trial-averaged eval, more
  bits at the low end) are part of the story — they make the rest credible.
- Don't bury the negative result. The v3 generation *hurt* the story; we
  diagnosed why and that diagnosis is the most interesting part of the
  architectural narrative.
- Don't show code. Show the equation, the data flow, the numbers, the plots.

---

## 2. Visual design language

The project's live web demo uses a deliberate **"2000s government website"**
aesthetic — hard corners, no rounded boxes, no shadows, navy + cream + black
+ government red, Arial body / Courier New mono. **Inherit this look in the
presentation.** It signals: this is a serious technical artifact, not a
startup pitch.

### Palette (use exactly these)

| role | hex | usage |
|---|---|---|
| **navy primary** | `#003366` | section bars, primary buttons, headings background |
| **navy dark** | `#002244` | navy borders / accents |
| **gov red / accent** | `#b23b34` | NeuroZip wins, callouts, the headline number |
| **gold accent** | `#ffcc33` | on-navy highlights, the brand letter in "NeuroZip" |
| **cream background** | `#f4f0e6` | page / slide background |
| **cream secondary** | `#e8e2d0` | sidebar / nav background |
| **white card** | `#ffffff` | content card surfaces |
| **black ink** | `#000000` | body text, hard rules |
| **muted gray** | `#888888` | secondary borders, captions |
| **link blue** | `#0033aa` | hyperlinks, with `underline` always |

### Typography

- **Headings:** `Arial, Verdana, "Helvetica Neue", sans-serif` — bold, uppercase
  for section bars, navy `#003366` background with white text.
- **Body:** same Arial stack, 13–15 px equivalent, color `#000`.
- **Numerals / data / code:** `"Courier New", Courier, monospace`. All metric
  values (bpp, ratios, percentages) render in Courier.
- **No web fonts.** Use system Arial and Courier New. The retro fidelity is
  the point.

### Layout rules

- **Hard corners only.** `border-radius: 0`. Everywhere.
- **No shadows.** `box-shadow: none`. Everywhere.
- **No transitions / animations** on hover. Static, sober.
- **Hard 1px black borders** around content cards. 2px navy borders around
  navigation/structural elements.
- **Section title bars:** navy `#003366` background, white text, uppercase,
  bold, 1px black bottom border. Looks like a form-section header on a 2002
  state-DMV website.
- **Tables:** ruled — `border: 1px solid #000` around the table, white cells,
  `#003366` header row with white text, alternating rows optional but light.
- **Links:** always underlined, `#0033aa` unvisited, `#551a8b` visited.
- **Buttons:** flat with `linear-gradient(#fafafa, #e2e2e2)`, 1px gray border;
  primary action button = solid navy with white text.

### Tonal "feel" notes

- Slides should feel like printed pages of a 2002 federal report on
  electroencephalographic data compression, *with one extremely modern chart
  per slide*. The contrast between sober gov-form chrome and a clean modern
  retrieval plot is the visual hook.
- Where a slide needs "NeuroZip" branded, render it as **Neuro**`<span style="color:#ffcc33">`**Zip**`</span>` on a navy header bar — the literal vault-title style from the demo's sidebar.
- Captions in Courier New, gray `#666`, lowercase.
- For "compression ratio" or hero numerics, render giant in Courier New
  (60–96px), `#b23b34` red.

---

## 3. The story arc (deck-level outline)

The deck must move through these beats in order. Each beat → one slide unless
noted. Numbers in brackets are the section of this brief where the supporting
material lives.

1. **Title.** *NeuroZip — task-aware EEG compression.* Subtitle: "the bits
   you keep are the bits that *mean* something." THINGS-EEG · sub-01 · v4. [§0]
2. **The problem.** EEG datasets are huge and growing; storage forces lossy
   compression; MSE-only codecs throw away the *meaning* (the part the dataset
   exists for). Reference the Haitao re-release already storing in fp16. [§4]
3. **The pitch (one sentence + diagram).** Same frozen model decides what to
   discard *and* lets you text-search what's kept. Show the data-flow
   diagram from §6. [§5, §6]
4. **What's preserved (the headline result).** At ~144× compression NeuroZip
   retains 29% top-5 image-prompt retrievability; fidelity at 210× retains
   19%. Visual: the big number "144×" / "+10 pp" hero. [§7]
5. **How it works (architecture).** The encoder → quantize → decoder →
   frozen judge `P` → CLIP loop, with the loss equation. Mark the three
   frozen blocks and the gradient path explicitly. [§6]
6. **The loss in one line.**
   `L = λ_rate · bits + λ_recon · ‖EEG − EEĜ‖² + λ_task · (1 − cos(P(EEĜ), CLIP_image(seen)))`
   Annotate each term with what it buys you. [§6]
7. **The three things to get right.** (a) Gradient flow through the frozen
   judge — `_grad_flow_assert` enforces this; (b) quantizer mismatch — train
   with `U(-0.5, 0.5)` noise, infer with `round`; (c) normalization stats
   round-trip. One slide each is overkill; one slide *total* with three short
   panels works. [§8]
8. **The numbers — projector (the judge).** Conv-only 2.5M params: 13.5% /
   37.5% / 50.5% top-1/5/10. **Attention head 6.0M params: 18.5% / 45.0% /
   65.0%.** ~+37% relative top-1. Show as a 2-row table. [§9.1]
9. **The numbers — codecs (v4 recommended).** The full results.md table for
   v4 fidelity vs v4 NeuroZip across low/med/high/xhigh. Highlight the
   NeuroZip rows in red. [§9.2]
10. **The money plot.** Rate–retrieval curve: top-5 (y) vs bpp (x), both
    families. NeuroZip's curve sits above fidelity at every tier. Asset:
    `demo/assets/rate_retrieval.png`. [§9.3]
11. **The asymmetric trade.** NeuroZip is *worse on MSE by ≤5%*, but *better
    on top-5 by +4 to +11 pp*. That asymmetry is the whole thesis. [§9.4]
12. **The circularity defense.** A separately trained EEG→concept classifier
    (different loss, different judge) confirms NeuroZip's win generalizes
    beyond the projector it was trained against. At v4, held-out classifier
    saturates near 100% for everything except fidelity at the most aggressive
    tier — which is exactly the signature of "fidelity at 210× actually lost
    decodable content." [§10]
13. **The v1 → v4 evolution (the negative-result slide).** v3 *hurt* the
    story; switching the codec to a ViT bottleneck collapsed the gap between
    fidelity and NeuroZip. Three mechanisms: (a) capacity leak — ViT
    bottleneck implicitly preserves ERP structure even under pure MSE; (b)
    judge saturation; (c) underconverged ViT codec in our training budget.
    v4 is the clean experiment: hold codec architecture fixed, vary only
    `lambda_task`. [§11]
14. **The CLIP gotcha.** The dataset's "ViT-B-32" features are LAION-2B, not
    OpenAI. Verified by cosine: LAION encoder → cos 0.98, OpenAI encoder →
    cos −0.06 (orthogonal). Live-inference text encoder *must* match.
    Small but critical. [§12]
15. **Demo / inference paths.** Two surfaces: Flask + `demo.html` (the
    gov-website page), and `notebook.ipynb`. Same modules; can't drift.
    [§13]
16. **Honest caveats.** One subject. Trial-averaged eval. NeuroZip uses
    slightly more bits at the most aggressive tier (fair framing: "NeuroZip
    at 144× beats fidelity at 210× by 10 pp"). [§14]
17. **What you'd buy (concrete storage example).** 1M labeled trial epochs:
    raw fp16 ≈ 30 GB → NeuroZip @ 144× ≈ 210 MB, retaining 29% top-5 and
    100% held-out classifier. [§9.5]
18. **Closing.** The contribution is *task-aware training*, not codec
    expressiveness. The general lesson: if your data exists to be *used*,
    compress it against the use, not against the L2 norm.

---

## 4. The problem (deeper)

THINGS-EEG epochs are 1-second visual-presentation trials, not long clinical
recordings. The storage pressure NeuroZip targets is **dataset-scale**:
millions of labeled trial epochs of brain↔image pairs. The Haitao999 release
re-stored EEG in float16 to halve size — direct evidence this storage
pressure is real and already being addressed crudely.

The standard answer ("use a lossy codec") is wrong for this regime, because
existing EEG codecs optimize MSE / PSNR. MSE is indifferent to whether the
parts of the signal it preserved are the parts you care about. For a
dataset whose *entire reason for existing* is brain↔stimulus alignment,
preserving alignment per bit is the right objective.

---

## 5. The pitch in one sentence

> The same frozen model that decides what to throw away during compression is
> what lets you text-search what you kept.

This is the single most important sentence in the deck. Lead with it on the
pitch slide.

---

## 6. How it works — architecture & data flow

### Data flow (verbatim from README)

```
              ┌─────────────────────────────────────────────────┐
              │           NeuroZip codec (trained)              │
EEG epoch ──► │  encoder ─► quantize ─► decoder ──► EEĜ        │
              └────────────────────┬────────────────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────┐
                │  projector P (frozen judge)      │
                │  EEĜ ──► CLIP-image embedding ε̂ │
                └──────────────────────────────────┘
                                   │
                ┌──────────────────┴────────────────────────────┐
                │  CLIP-image embedding of the seen image  ε*   │  (frozen, no_grad)
                └─────────────────────────────────────────────-─┘
                                   │
                       L_task = 1 − cos(ε̂, ε*)
                              ──+──
   L  =  λ_rate · bits  +  λ_recon · ‖EEG − EEĜ‖²  +  λ_task · L_task
```

### Loss equation (use as a slide centerpiece)

```
L = λ_rate · bits(latent | prior)
  + λ_recon · ‖EEG − EEĜ‖²
  + λ_task · (1 − cos(P(EEĜ), CLIP_image(seen_image)))
```

Term-by-term:
- **Rate term** — bits estimated under a factorized Laplace prior with
  learnable per-channel scale. Buys you compression.
- **Recon term** — standard MSE between input EEG and reconstructed EEĜ.
  Anchors the codec; without it the task term wins and the EEG looks like
  garbage even if it decodes correctly.
- **Task term** — cosine distance between the *reconstructed* EEG's CLIP
  embedding (via frozen projector `P`) and the *frozen* CLIP image embedding
  of what the subject actually saw. Buys you retrievability.

### Tensor shapes

- Input EEG: `(B, 63, 250)` — 63 channels × 250 timesteps (1 s @ 250 Hz).
- Latent: `(B, 32 channels × 32 timesteps) = 1024 integer symbols` per epoch.
- Embedding space: 512-dim, CLIP ViT-B-32 LAION-2B variant.

### Components

- **Codec (`codec.py`):** 1D-conv autoencoder over time. Mirror-pad 250→256,
  downsample /2/2/2 to 32, decoder mirrors. Optional ViT bottleneck (`n_attn>0`).
  Default: conv-only, ~0.75 M params. Factorized Laplace prior for rate.
  Uniform-noise quantizer at train; integer `round` at inference.
- **Projector P (`clip_proj.py`):** depthwise temporal conv → spatial mixer →
  4-stage conv tower → optional [CLS] + transformer head → L2-norm.
  Recommended `n_attn=2`, ~6.0 M params. Stage-1 trained with symmetric
  InfoNCE against precomputed CLIP image features. Frozen for all later
  stages.
- **Held-out classifier:** separate EEG→concept classifier (softmax over 200
  concepts), trained on a disjoint slice of the test repetitions. Never sees
  codec output. Used purely as an independent judge in the circularity
  defense.

---

## 7. Headline result

> **At ~144× compression, NeuroZip retains 29% top-5 image-prompt
> retrievability. The fidelity baseline at 210× retains 19%. NeuroZip beats
> fidelity at every comparable rate tier while staying within ±5% on raw
> MSE.**

Hero numbers for the big-number slide:
- **144×** compression
- **+10 pp** top-5 lift at this tier
- **100%** held-out classifier accuracy (independent judge agrees)

---

## 8. The three easy-to-get-wrong spots

These are the "from the trenches" details. One slide is enough.

1. **Gradient flow through the frozen judge.** `P` is `requires_grad_(False)`
   and the *target* embedding `CLIP_image(seen)` is `no_grad`. But `P(EEĜ)`
   itself must NOT be wrapped in `no_grad` — the decoder still needs gradient
   *through* `P`. `train.py:_grad_flow_assert()` runs once at stage-3 startup
   and asserts (a) decoder gets non-zero grad from a task-only backward, (b)
   judge params get zero grad. Skipping this is the most common silent
   failure mode.
2. **Quantizer mismatch.** Train with `Uniform(-0.5, 0.5)` noise as a
   differentiable proxy for integer rounding. Inference uses `torch.round`.
   Forget the noise at training time and `round()` breaks the decoder at
   inference.
3. **Normalization stats round-trip.** EEG is per-channel z-normalized in
   `data.py`. Mean/std cached at `data/norm_stats.pt`. Needed (a) to invert
   codec output to microvolts for fidelity reporting; (b) so bpp is bits per
   *normalized* sample, comparable across configs and against the fp16
   storage baseline.

---

## 9. The numbers

### 9.1 Projector (the frozen judge)

| projector | params | top-1 | top-5 | top-10 |
|---|---:|---:|---:|---:|
| conv-only (AvgPool + MLP head) | 2.5 M | 13.5% | 37.5% | 50.5% |
| **conv + [CLS] + 2 attn blocks** | **6.0 M** | **18.5%** | **45.0%** | **65.0%** |

Chance on 200-way retrieval: 0.5% / 2.5% / 5.0%. Attention head clears chance
by 27× / 18× / 13× respectively. The +37% relative top-1 lift from the
attention head is the dominant projector design choice.

### 9.2 Codecs — v4 recommended generation

**Fidelity baseline (MSE + rate only):**

| codec | bpp ↓ | ratio ↑ | MSE | top-1 | top-5 | top-10 | held-out top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fidelity_v4_low` | 0.076 | 210× | 0.0401 | 5.5% | 19.0% | 29.0% | 88.0% |
| `fidelity_v4_med` | 0.154 | 104× | 0.0298 | 8.0% | 26.5% | 38.0% | 99.5% |
| `fidelity_v4_high` | 0.209 | 76× | 0.0231 | 8.0% | 26.5% | 40.5% | 100.0% |
| `fidelity_v4_xhigh` | 0.248 | 64× | 0.0213 | 9.5% | 31.5% | 43.5% | 100.0% |

**NeuroZip (MSE + rate + task):**

| codec | bpp ↓ | ratio ↑ | MSE | top-1 | top-5 | top-10 | held-out top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `neurozip_v4_low` | 0.111 | 144× | 0.0359 | **10.5%** | **29.0%** | **42.5%** | **100.0%** |
| `neurozip_v4_med` | 0.190 | 84× | 0.0251 | **10.5%** | **32.5%** | **47.0%** | **100.0%** |
| `neurozip_v4_high` | 0.222 | 72× | 0.0234 | **14.0%** | **37.5%** | **52.0%** | **100.0%** |
| `neurozip_v4_xhigh` | 0.250 | 64× | 0.0223 | **14.5%** | **35.5%** | **50.0%** | **100.0%** |

**Side-by-side at matched tiers:**

| tier | NZ bpp / fid bpp | ratio comparison | top-5 lift |
|---|---|---|---|
| low | 0.111 / 0.076 | 144× / 210× | NZ +10.0 pp |
| med | 0.190 / 0.154 | 84× / 104× | NZ +6.0 pp |
| high | 0.222 / 0.209 | 72× / 76× | NZ +11.0 pp |
| xhigh | 0.250 / 0.248 | 64× / 64× | NZ +4.0 pp |

### 9.3 The money plot

`demo/assets/rate_retrieval.png` — top-5 retrieval (y) vs bpp (x), both
families. NeuroZip's curve sits above fidelity at every tier. If the
presentation can embed a single image, this is it.

### 9.4 The asymmetric trade

- Intra-v4 fidelity-vs-NeuroZip **MSE gap** is small (typically ±5%).
- **Top-5 lift** is +4 to +11 pp (~+13% to +42% relative).
- **NeuroZip's worst MSE deficit (~5%) is more than paid for by its best
  top-5 advantage (+11 pp / ~+42% relative).** The trade is heavily
  asymmetric in NeuroZip's favor.

### 9.5 Per-codec storage in human terms

Raw fp16 epoch = 63 × 250 = 15,750 samples = **31.5 kB**.

| codec | bytes per epoch | ratio |
|---|---:|---:|
| raw fp16 | 31,500 | 1× |
| neurozip_v4_xhigh | ~492 | 64× |
| fidelity_v4_high | ~412 | 76× |
| neurozip_v4_med | ~374 | 84× |
| fidelity_v4_med | ~303 | 104× |
| neurozip_v4_low | ~219 | 144× |
| fidelity_v4_low | ~150 | 210× |

**1M-epoch corpus (realistic THINGS-EEG-scale):**
raw fp16 = ~30 GB → NeuroZip @ 144× = ~210 MB, retaining 29% top-5
retrievability and 100% held-out classifier accuracy.

---

## 10. The circularity defense

The codec is trained with `P` + CLIP in the loop. If we *only* evaluated
retrieval with `P` + CLIP, "great numbers against your own judge" is a fair
takedown. So:

- `train.py classifier` trains a **separate** EEG→concept classifier on 60
  of the 80 test repetitions per concept (averaging blocks of 10 reps as
  inputs for SNR), then scores on each codec's decompressed test EEG. It
  never sees codec output during its own training.
- Different judge: softmax over 200 concepts vs contrastive against CLIP.
  Different loss, different objective.
- With the v4 attention judge, this classifier is so strong it saturates
  near 99–100% for everything except `fidelity_v4_low` at 210× (88%).

**Reading.** Both methods preserve enough decodable content for an
independent classifier to identify the concept, *except* fidelity at the
most aggressive tier — where the held-out judge confirms that fidelity has
actually lost content. NeuroZip at 144× is at 100%. Method differentiation
moves to the harder projector-based retrieval metric, which is the more
interesting one anyway because it composes with text queries.

---

## 11. The v1 → v4 evolution (the negative-result narrative)

This is the slide that earns the audience's trust.

| version | codec | projector | classifier | story |
|---|---|---|---|---|
| **v1** | conv-only (0.75 M) | conv-only (2.5 M) | conv-only | first runnable; NeuroZip dominates everywhere |
| **v2** | ViT, n_attn=2 (4.3 M) | conv-only (2.5 M) | conv-only | bigger codec, slight gains, longer training |
| **v3** | ViT, n_attn=2 (4.3 M) | **ViT (6.0 M)** | ViT | stronger judge; NeuroZip win **shrank** |
| **v4** | conv-only (0.75 M) | ViT (6.0 M) | ViT | clean baseline + strong judge; recommended |

### Why v3 hurt the story (three reinforcing mechanisms)

1. **Capacity leak into the baseline.** A ViT bottleneck captures long-range
   temporal structure even under pure MSE. Some of that structure is exactly
   the ERP timing the projector keys off. So the fidelity baseline was
   implicitly preserving semantic content without ever seeing the task loss.
2. **Judge saturation.** Stronger judge → higher floor for *everyone*. The
   held-out classifier went from 53–85% (fidelity, v1 conv judge) to 99–100%
   (fidelity, v2 attention judge). Differentiation collapsed.
3. **Underconverged ViT codec.** The 4.3 M-param codec needed more than 30
   epochs to fully train; within our budget its MSE was actually *worse*
   than the 0.75 M conv codec. NeuroZip warm-started from undertrained
   fidelity codecs and inherited the deficit.

### Why v4 is the recommendation

The contribution is **task-aware training**, not codec expressiveness. The
cleanest demonstration: hold codec architecture fixed across fidelity and
NeuroZip, vary only `λ_task`.

- **Codec = conv-only.** Fidelity baseline can't ride architectural capacity
  to fake semantic preservation.
- **Projector = attention.** Task gradient is sharp; retrieval at inference
  is strong. The judge upgrade doesn't pollute the baseline because the
  baseline doesn't see the projector during training.
- **Classifier = attention.** Circularity defense uses a strong independent
  judge that isn't trivially fooled.

The clean experimental claim under v4: *"with the codec held fixed, the task
loss preserves more CLIP-decodable content per bit than MSE+rate alone."*

---

## 12. The CLIP gotcha

The dataset `Haitao999/things-eeg` ships precomputed CLIP image and text
features at `Preprocessed_data_250Hz_whiten/ViT-B-32_features_*.pt`. The
filenames say `ViT-B-32`, but **the features are not from the standard OpenAI
weights** — they're from the LAION-2B variant
(`open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')`).

Verified by direct cosine: an image encoded by LAION-2B's encoder matches
the dataset's precomputed image feature at **cos = 0.98**. OpenAI's encoder
gives **cos = −0.06** (orthogonal — different space entirely).

This matters because `P` is trained against the dataset's image features, so
its output lives in LAION-2B space. The live-inference text encoder in
`serve.py` *must* use the matching variant or free-text retrieval breaks
silently (CLIP query lands in a different subspace; retrieval becomes random).

Small detail, presentation-worthy because it's the kind of bug that would
have made all the numbers wrong for a non-obvious reason.

---

## 13. Inference / demo

Two viewing surfaces, both backed by the same checkpoints, sharing
`data.py` / `clip_proj.py` / `codec.py` so the notebook and the server
cannot drift apart.

- **`serve.py` + `demo.html`** — Flask backend; live CLIP-text encoding via
  the LAION-2B encoder; on-demand codec reconstruction; server-rendered
  matplotlib figures returned as base64 PNGs. The demo page is the
  gov-website look this brief inherits.
- **`notebook.ipynb`** — standalone Jupyter; auto-detects which codec
  generation is on disk (prefers v4). 20 cells: setup → metrics table →
  rate–retrieval plot → reconstruction viewer → free-text retrieval →
  per-channel MSE + entropy histograms.

If the presentation includes a "try it" CTA, point to `./serve.sh` and the
default `http://<host>:8011/`.

---

## 14. Honest caveats

Include all of these. They make the rest credible.

- **One subject.** sub-01. The pipeline is subject-agnostic — change a
  single flag in `data.py` to run any of the 10 subjects — but reported
  numbers are for one.
- **Trial-averaged eval.** Each of 200 test concepts has 80 EEG reps;
  retrieval is computed on the average of all 80. Single-trial retrieval is
  much noisier; trial-averaging is the standard THINGS-EEG protocol.
- **NeuroZip uses more bits at the low end** (0.111 vs 0.076 bpp). The fair
  framing is *"NeuroZip at 144× beats fidelity at 210× by 10 pp"*, not
  *"matched bpp at the low tier."* At high and xhigh, bpps are essentially
  matched.
- **Strong-judge retrieval is the main metric** (intentionally — harder and
  more compositional). If you prefer the held-out classifier as sole judge,
  differentiation is much smaller because both methods saturate.
- **CLIP gotcha.** Wrong CLIP variant silently breaks retrieval. See §12.
- **Epochs are 1-second visual trials**, not multi-hour clinical
  recordings. Generalizing to long continuous EEG would require a streaming
  codec; that's future work.

---

## 15. Glossary (use in a backup slide, or as on-slide tooltips)

- **bpp.** Bits per EEG sample. Raw fp16 = 16.0 bpp. `bpp = total compressed
  bits per epoch ÷ 15,750`. Lower is better.
- **ratio.** `16 / bpp`. Higher is better.
- **MSE.** Mean squared error between raw and decompressed EEG on per-channel
  z-normalized signals (mean 0, std 1 per channel). Squared fraction of one
  channel's std; multiply by `data/norm_stats.pt` to get squared microvolts.
- **top-k.** For each of 200 test concepts: trial-average → compress →
  decompress → project to CLIP space → rank 200 CLIP image embeddings by
  cosine to that vector → "hit" if gold concept's image lands in top-k.
  Averaged across concepts. Chance = k/200. Ceiling (raw EEG, attention
  projector) = 18.5% / 45.0% / 65.0%.
- **held-out top-1.** Top-1 of the independently trained EEG→concept
  classifier on each codec's decompressed EEG. The circularity defense.
- **gold rank.** Where the correct concept ranked, out of 200. Lower better.
- **CLIP cos.** Cosine between your free-text query (CLIP-text) and the
  closest known test concept (also CLIP-text). Tells you if your query is
  on-vocabulary. >0.7 well-matched; ~0.3 a stretch.
- **bits/symbol.** Entropy of one quantized latent integer under the learned
  Laplace prior. `bpp = bits/symbol × 1024 ÷ 15,750`.
- **tier (low/med/high/xhigh).** Four rate-distortion operating points
  tuned via `λ_rate` in the loss. NeuroZip uses slightly higher `λ_rate` per
  tier so the task term doesn't drift bpp up.
- **ERP.** Event-related potential — a stereotyped brain response timed to a
  stimulus. P100 / N170 / P300 are particular components at characteristic
  latencies. The attention projector keys off these.

---

## 16. Setup table (use on a "what we ran on" slide)

| | value |
|---|---|
| Subject | `sub-01` (one of 10 in the dataset) |
| EEG | 63 channels, 250 Hz, 250-sample epochs (1 s post-stimulus + 0.2 s baseline) |
| Train epochs | 16,540 image-trials × 4 reps = 66,160 |
| Test concepts | 200 (disjoint from train) |
| Test repetitions | 80 per concept (16,000 epochs total) |
| Eval input | trial-averaged across all 80 reps per concept |
| CLIP variant | ViT-B/32 / LAION-2B (matches dataset's precomputed features) |
| Hardware | single RTX 4090 |
| Train wall-clock | ~45 min for the full v4 sweep |

---

## 17. Hyperparameters (back-pocket reference)

| tier | fidelity λ_rate | NeuroZip λ_rate | epochs |
|---|---:|---:|---:|
| low | 0.05 | 0.07 | 15 |
| med | 0.01 | 0.015 | 15 |
| high | 0.002 | 0.003 | 15 |
| xhigh | 0.0003 | 0.0005 | 15 |

`λ_task = 3.0` for all NeuroZip codecs; `λ_recon = 1.0` for all.

---

## 18. Per-query examples (good for a "see it work" slide)

Concepts where NeuroZip's lead over fidelity is largest (full v4 sweep,
top-5 image-prompt retrieval; "✓" = gold concept in top-5):

| query | NeuroZip ✓ | fidelity ✓ |
|---|---|---|
| robot | 3 of 4 tiers | 0 of 4 tiers |
| ostrich | 3 of 4 tiers | 0 of 4 tiers |
| unicycle | 3 of 4 tiers | 1 of 4 tiers |
| suit | 3 of 4 tiers | 1 of 4 tiers |
| submarine | 3 of 4 tiers | 1 of 4 tiers |

---

## 19. Visual assets you can pull in

- **`demo/assets/rate_retrieval.png`** — the money plot. Top-5 vs bpp,
  NeuroZip vs fidelity. Use it.
- **`demo/assets/summary.json`** — featured-concept list, full numerics.
- The data-flow ASCII diagram in §6 — render as a real diagram in the
  presentation's typography (Arial nodes, navy borders, courier labels).
- The v4 results tables in §9.2 — render as ruled gov-style tables, NeuroZip
  rows highlighted in red `#b23b34`.
- For the "what you'd buy" slide — a two-bar comparison: "30 GB raw fp16" vs
  "210 MB NeuroZip @ 144×", with the retention numbers underneath. Make it
  feel like a 2002 federal-agency cost-savings infographic.

---

## 20. Final design reminders

- **Hard corners. No shadows. No gradients other than the gov-button
  pseudo-bevel.**
- **Navy `#003366` section bars** with white uppercase Arial bold.
- **Red `#b23b34` only for NeuroZip wins** and headline numerics. Don't
  spray it.
- **Courier New for every number on every slide.** Even small ones in
  captions.
- **Underline every link.** It's 2002.
- The one allowed modernity: the rate-retrieval plot. Render it cleanly,
  let it sit on a white card with a 1px black border, navy title bar above.
  The contrast with the gov-form chrome around it *is* the visual story.

When in doubt: imagine a printed PDF report from the
National Institute of Brain-Data Compression, circa 2002, with one
shockingly modern chart per page. Build that.
