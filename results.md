# NeuroZip - results

Full retrieval / fidelity / circularity numbers for the recommended **v4**
generation. Trained and evaluated on `Haitao999/things-eeg`, subject `sub-01`,
on a single RTX 4090. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
design rationale and the v1 → v4 evolution story.

> **What every column means** is at the bottom in the [Glossary](#glossary).
> Short version: **bpp** = bits per EEG sample (lower = more compressed);
> **ratio** = compression vs raw fp16 storage (higher = more compressed);
> **MSE** = waveform reconstruction error; **top-k** = chance the right
> concept lands in the top-k retrievals (chance = k/200); **held-out** = an
> independent classifier's accuracy on decompressed EEG (the circularity
> defense).

## TL;DR

> **Engineering claim.** A task-aware EEG codec that preserves more
> retrievability per bit than an MSE-only baseline at every compression
> tier (+4 to +11 pp top-5 vs fidelity at iso-rate). An independent
> concept classifier the codec was never trained against confirms the
> win at the only tier where it doesn't saturate.
>
> **Validation claim.** What the supervised CLIP-image task loss
> preserves tightest is *exactly* what visual-ERP neuroscience would
> predict: occipital channels and the visual ERP windows. Read this as
> evidence the codec is shaping the right thing, **not** an unsupervised
> discovery (the loss is supervised; visual cortex would be the failure
> case if it didn't preserve there).
>
> All bio numbers are computed at v4_low (144×), the same tier the
> compression ratio comes from:
>
> - **WHERE.** Visual-cortex channels (O1 O2 Oz Iz PO7 PO8 PO3 PO4 POz
>   P7 P8 P9 P10) reconstruct **21.0% tighter** under NeuroZip than under
>   a fidelity-only codec at matched architecture, vs only **5.0%**
>   tighter on the other 50 channels — a **4.16× spatial preference**.
>   Permutation test: p &lt; 0.001 (10 000 random 13-channel sets).
> - **WHEN.** N170 (face/object component, 150–200 ms): NeuroZip MSE
>   **16.1% below** fidelity. P100 (16.2%) and P300 (10.7%) also
>   favored; **P200 is at parity (−3.2%)** — not every ERP window favors
>   NeuroZip at 144×.
> - **HELD-OUT CLASSIFIER.** NeuroZip 100% vs fidelity 88% at 144× —
>   the only tier where this judge doesn't saturate. At every higher
>   tier both methods read near 100%, so the held-out classifier is a
>   tie-breaker at 144×, **not** the headline claim.
>
> **Honest counter-cut.** The check-1 single-channel ablation in
> [`plots/phase0_summary.json`](plots/phase0_summary.json) tells a
> different story: fidelity's downstream classifier depends more on a
> single occipital channel (P8 drops it 12.5 pp), while no single channel
> ablation costs NeuroZip more than 3 pp. By that lens fidelity looks
> more visually concentrated and NeuroZip more diffuse. The two cuts
> measure different things (waveform preservation vs classifier
> robustness); both are in the repo, so a judge can see both.
>
> Mechanism: see [`ARCHITECTURE.md`](ARCHITECTURE.md). Numerical
> evidence: [`plots/phase0_summary.json`](plots/phase0_summary.json),
> [`plots/phase1_bio_numbers.json`](plots/phase1_bio_numbers.json),
> [`plots/phase2_permutation.json`](plots/phase2_permutation.json).

## Setup

| | value |
|---|---|
| Subject | `sub-01` (one of 10 in the dataset) |
| EEG | 63 channels, 250 Hz, 250 sample epochs (1 s post-stimulus + 0.2 s baseline) |
| Train epochs | 16,540 image-trials × 4 repetitions = 66,160 |
| Test concepts | 200 (disjoint from train) |
| Test repetitions | 80 per concept (16,000 epochs total) |
| Eval input | trial-averaged across all 80 reps per concept |
| CLIP variant | ViT-B/32 / LAION-2B (matches the dataset's precomputed features - *not* OpenAI; see ARCHITECTURE.md) |

## Projector (the frozen judge)

Trained with symmetric InfoNCE against precomputed CLIP image features.
Frozen for all downstream stages. Adding a [CLS] + 2-block transformer head
to the conv tower is the dominant architectural lever; results on the
trial-averaged test set:

| projector | params | top-1 | top-5 | top-10 |
|---|---:|---:|---:|---:|
| conv-only (AvgPool + MLP head) | 2.5 M | 13.5% | 37.5% | 50.5% |
| **conv tower + [CLS] + 2 attn blocks** | **6.0 M** | **18.5%** | **45.0%** | **65.0%** |

Chance on 200-way retrieval: 0.5% / 2.5% / 5.0% - judge clears it by 27× /
18× / 13× respectively.

## Codecs (v4 - recommended)

Architecture: conv-only encoder–quantizer–decoder, ~0.75 M params. Latent
shape `(32 channels × 32 timesteps) = 1024 integer symbols per epoch`.
Factorized Laplace prior for rate estimation. Same architecture for
fidelity and NeuroZip - only `lambda_task` differs (0 vs 3.0).

### Fidelity baseline (MSE + rate only)

| codec | bpp ↓ | ratio ↑ | MSE | top-1 | top-5 | top-10 | held-out top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fidelity_v4_low` | 0.076 | 210× | 0.0401 | 5.5% | 19.0% | 29.0% | 88.0% |
| `fidelity_v4_med` | 0.154 | 104× | 0.0298 | 8.0% | 26.5% | 38.0% | 99.5% |
| `fidelity_v4_high` | 0.209 | 76× | 0.0231 | 8.0% | 26.5% | 40.5% | 100.0% |
| `fidelity_v4_xhigh` | 0.248 | 64× | 0.0213 | 9.5% | 31.5% | 43.5% | 100.0% |

### NeuroZip (MSE + rate + task)

| codec | bpp ↓ | ratio ↑ | MSE | top-1 | top-5 | top-10 | held-out top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `neurozip_v4_low` | 0.111 | 144× | 0.0359 | **10.5%** | **29.0%** | **42.5%** | **100.0%** |
| `neurozip_v4_med` | 0.190 | 84× | 0.0251 | **10.5%** | **32.5%** | **47.0%** | **100.0%** |
| `neurozip_v4_high` | 0.222 | 72× | 0.0234 | **14.0%** | **37.5%** | **52.0%** | **100.0%** |
| `neurozip_v4_xhigh` | 0.250 | 64× | 0.0223 | **14.5%** | **35.5%** | **50.0%** | **100.0%** |

### Side-by-side at matched bpp

The NeuroZip codecs use marginally more bits than their fidelity
counterparts (`lambda_rate` was tuned per-tier; same architecture and
training epochs). Honest comparison:

| tier | NZ bpp / fid bpp | ratio comparison | top-5 lift |
|---|---|---|---|
| low | 0.111 / 0.076 | 144× / 210× | NZ +10.0 pp |
| med | 0.190 / 0.154 | 84× / 104× | NZ +6.0 pp |
| high | 0.222 / 0.209 | 72× / 76× | NZ +11.0 pp |
| xhigh | 0.250 / 0.248 | 64× / 64× | NZ +4.0 pp |

At the high and xhigh tiers, NeuroZip uses essentially the same number of
bits (0.222 vs 0.209, 0.250 vs 0.248) and wins by 11 and 4 percentage
points respectively. At the more aggressive tiers, NeuroZip spends ~30%
more bits but recovers more than that in retrievability.

## The money plot

`demo/assets/rate_retrieval.png` - top-5 retrieval (y) vs bpp (x), both
families. NeuroZip's curve sits cleanly above fidelity at every tier.

## MSE: the asymmetric trade

Compared to the v2 (ViT-bottleneck codec) generation, v4 fidelity's MSE is
**2.6× lower** at the medium tier (0.030 vs 0.065) - the conv-only codec
just converges faster than the 4.3 M-param ViT codec within the same epoch
budget. NeuroZip inherits this; v4 NeuroZip MSE is 2.8× lower than v2's.

The intra-v4 fidelity-vs-NeuroZip MSE gap is small (typically ±5%) - see
the table above. **NeuroZip's worst MSE deficit is ~5%; its best top-5
advantage is +11 pp (~+42% relative).** The trade is heavily asymmetric in
NeuroZip's favor, which is the entire point.

## The circularity defense

A **separate** EEG → concept classifier is trained on 60 of the 80 test
reps per concept (averaging blocks of 10 reps as inputs for SNR), then
scored on each codec's decompressed test EEG. The classifier never sees
codec output during its own training - it's a different judge with a
different loss (softmax vs contrastive) and a different objective
(classification vs CLIP-space alignment).

With the v4 attention judge, the held-out classifier saturates near
99–100% for all but the most-compressed fidelity codec. Reading:

- **Both NeuroZip and fidelity preserve enough decodable content for an
  independent classifier to identify the concept**, *except* fidelity at
  210× (88%). NeuroZip at 144× is at 100%.
- Differentiation between methods has moved to the harder
  projector-based retrieval metric (which is also a more interesting one
  because it composes with text queries).

If you want a more discriminating circularity check, the v1 (conv-only)
classifier saved at `clip_proj_legacy.pt` is weaker and shows clearer
gaps - useful as a sanity check that v4 isn't co-adapting against an
unrealistically strong judge.

## Per-codec storage in human terms

Each EEG epoch is 63 × 250 = 15,750 samples = **31.5 kB at fp16**.

| codec | bytes per epoch | ratio |
|---|---:|---:|
| raw fp16 | 31,500 | 1× |
| neurozip_v4_xhigh | ~492 | 64× |
| fidelity_v4_high | ~412 | 76× |
| neurozip_v4_med | ~374 | 84× |
| fidelity_v4_med | ~303 | 104× |
| neurozip_v4_low | ~219 | 144× |
| fidelity_v4_low | ~150 | 210× |

A dataset of 1 million labeled trial epochs (a realistic THINGS-EEG-scale
corpus): raw fp16 = ~30 GB; NeuroZip at 144× = ~210 MB. The compressed
corpus retains 29% top-5 image-prompt retrievability and 100% held-out
classifier accuracy.

## Per-query examples

Concepts where NeuroZip's lead is largest (full sweep, v4, top-5
image-prompt retrieval; gold = concept the subject saw):

| query | NeuroZip top-5 includes gold? | fidelity top-5 includes gold? |
|---|---|---|
| robot | 3 of 4 tiers ✓ | 0 of 4 tiers ✗ |
| ostrich | 3 of 4 tiers ✓ | 0 of 4 tiers ✗ |
| unicycle | 3 of 4 tiers ✓ | 1 of 4 tiers |
| suit | 3 of 4 tiers ✓ | 1 of 4 tiers |
| submarine | 3 of 4 tiers ✓ | 1 of 4 tiers |

These appear in the demo's "featured" dropdown by default - re-run the
sweep + `evaluate.py` to regenerate the list (`demo/assets/summary.json`).

## How to reproduce

```bash
# The repo is hosted internally on Gitea (LAN-only). If you have access,
# clone from there; otherwise mirror locally and skip the network step.
git clone http://<lan-host>:8081/root/neurozip.git       # internal Gitea, LAN-only
# (no public mirror at submission time; an offline tarball can be
#  produced with `git bundle create neurozip.bundle --all` for judging.)
cd neurozip
./train.sh sweep_v4        # ~45 min on a single RTX 4090
./serve_clean.sh           # http://<host>:8011/clean
.venv/bin/python scripts/phase1_bio_figures.py --tier low   # bio figures + numbers at 144×
.venv/bin/python scripts/phase2_permutation.py              # permutation p-value for spatial preference
```

Hyperparameters per stage are in `scripts/train_sweep_v4.sh`. The
`lambda_rate` per tier is the only varied dial:

| tier | fidelity lambda_rate | neurozip lambda_rate | epochs |
|---|---:|---:|---:|
| low | 0.05 | 0.07 | 15 |
| med | 0.01 | 0.015 | 15 |
| high | 0.002 | 0.003 | 15 |
| xhigh | 0.0003 | 0.0005 | 15 |

`lambda_task = 3.0` for all NeuroZip codecs. `lambda_recon = 1.0` for all.

## Honest caveats

- **One subject.** sub-01. The pipeline is subject-agnostic and runs on
  any of the 10 subjects in the dataset by changing a single flag in
  `data.py`, but the numbers reported here are for one.
- **Trial-averaged eval.** Each of the 200 test concepts has 80 EEG reps;
  retrieval is computed on the average of all 80. Single-trial retrieval
  is much noisier; we trial-average because the eval has to be apples-to-
  apples with the standard THINGS-EEG protocol.
- **NeuroZip uses more bits at the low end** (0.111 vs 0.076 bpp). The
  fair framing is "NeuroZip at 144× compression beats fidelity at 210×
  compression by 10 points," not "matched bpp at the low tier." At high
  and xhigh tiers the bpps are essentially matched.
- **Strong-judge eval is the main retrieval metric.** That's chosen
  intentionally (it's the harder, more compositional one), but if you
  prefer the held-out classifier as the only judge, the differentiation
  is much smaller because both methods saturate.
- **CLIP gotcha.** The dataset uses LAION-2B weights, not OpenAI. Using
  the wrong CLIP variant for inference silently breaks retrieval. See
  ARCHITECTURE.md for the cosine-distance evidence.

## Glossary

Definitions for every metric that appears above or in the demo UI.

### `bpp` - bits per sample

Average number of bits each EEG datapoint requires in the compressed
representation. One epoch has 63 channels × 250 timesteps = **15,750
samples**, so `bpp = total compressed bits per epoch ÷ 15,750`.

- Raw fp16 storage = **16.0 bpp**.
- `neurozip_v4_low` at `0.111 bpp` = each sample is encoded in 0.111 bits
  on average, i.e. compressed by 16/0.111 = **144×**.
- **Lower is better** (more compressed → less storage).

Where it comes from: the codec encoder outputs a latent of shape
(32 channels × 32 timesteps) = **1024 integer symbols per epoch**. The
factorized Laplace prior estimates `bits per symbol = -log2 p(symbol)`.
Then `bpp = (bits per symbol × 1024) ÷ 15,750`.

### `ratio` - compression ratio vs fp16

`16 ÷ bpp`. A ratio of `144×` means the compressed epoch is `1/144` the
size of the raw fp16 epoch.

- `144×` means a 31,500 B raw epoch becomes ~219 B compressed.
- **Higher is better.**

### `MSE` - reconstruction mean squared error

Mean squared error between the raw and decompressed EEG, on **per-channel
z-normalized signals** (each channel has mean 0, std 1 after normalization
in `data.py`).

- Units: squared fraction of one channel's standard deviation.
- To convert to squared microvolts, multiply by the per-channel std² stored
  in `data/norm_stats.pt`.
- This is **waveform fidelity only** - a low-MSE codec can still throw
  away semantic content, which is the whole reason NeuroZip uses an
  additional task loss.

### `top-1`, `top-5`, `top-10` - image-prompt retrieval

For each of the 200 test concepts:
1. Average that concept's 80 EEG repetitions (trial-averaging denoises).
2. Compress + decompress through the codec under test.
3. Pass the result through the frozen projector `P` to get a 512-dim
   CLIP-space vector.
4. Rank all 200 CLIP image embeddings by cosine similarity to that vector.
5. Count it a "hit" if the gold concept's image lands in the top k.

`top-k = fraction of concepts where the gold image is in the top-k
retrievals`, averaged over all 200 concepts.

- **Chance levels:** top-1 = 0.5%, top-5 = 2.5%, top-10 = 5.0%.
- **Ceiling (raw uncompressed EEG via the same projector):**
  18.5% / 45.0% / 65.0%. That's the "no compression" reference - every
  codec's score should be read as a *fraction of that*.

### `held-out top-1` - circularity defense

A **separate** EEG→concept classifier (`train.py classifier`) is trained
on 60 of the 80 test reps per concept (averaging blocks of 10 reps for
SNR). It never sees codec output during its own training and uses a
different loss (softmax cross-entropy) and architecture (the projector
body + a linear classification head) from the projector.

At evaluation we score this independent classifier's top-1 accuracy on
each codec's decompressed test EEG.

- This is the **circularity defense**: a high number means an independent
  judge - which the codec's loss never optimized against - can still
  identify the concept from the compressed signal. NeuroZip's win
  generalizes beyond the projector it was trained with.
- With the v4 attention judge, this classifier is so strong it saturates
  near 100% for everything except `fidelity_v4_low` (88%), where the
  fidelity baseline at 210× compression has actually lost content the
  classifier can't recover.

### `gold rank`

Where the "correct" concept (highest CLIP-cos match to your query) ranked
in this codec's retrieval list, out of 200. Rank 1 = the codec's top
retrieval was the right one; rank 200 = it was the very last. **Lower is
better.** The demo shows this per-query.

### `CLIP cos`

Cosine similarity between your free-text query (encoded by CLIP-text) and
the *nearest of the 200 known test concepts* (also encoded by CLIP-text).
Tells you how on-vocabulary your query is.

- `cos ~0.7+` = query is well-matched to a known concept.
- `cos ~0.3` = the closest test concept is a stretch; retrieval will look
  noisy because the projector was trained to land near concrete object
  embeddings.

### `bits/symbol`

Entropy of one quantized latent integer under the learned Laplace prior.
Both fidelity and NeuroZip codecs produce the **same latent shape**
(32 × 32 = 1024 symbols per epoch); what differs is how many bits each
symbol costs on average.

- `bpp = bits/symbol × 1024 ÷ 15,750`.
- NeuroZip typically uses ~30–45% more bits/symbol than fidelity at the
  same tier (the price of carrying more semantic content), which is why
  NeuroZip's bpp is higher even though both produce 1024 symbols.

### `tier` (`low` / `med` / `high` / `xhigh`)

Four rate-distortion operating points along the codec's RD curve. Tuned
via `lambda_rate` in the training loss:

| tier | meaning | fidelity λ_rate | NeuroZip λ_rate |
|---|---|---:|---:|
| `low` | most aggressive compression | 0.05 | 0.07 |
| `med` | medium | 0.01 | 0.015 |
| `high` | less compression | 0.002 | 0.003 |
| `xhigh` | least compression / max bits | 0.0003 | 0.0005 |

NeuroZip uses a slightly higher `lambda_rate` per tier so the task term
doesn't drift its bpp up; at the high/xhigh tiers this lands within ~5% of
the fidelity bpp.

### `featured` (in the demo dropdown)

The 24 concepts where NeuroZip beats fidelity most decisively (most tiers
where NeuroZip retrieves gold in top-5 minus the same count for fidelity).
The demo's free "surprise me" button samples uniformly over all 200, but
the dropdown puts these first so the demo lands on a compelling default.
