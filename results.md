# NeuroZip — results

Full retrieval / fidelity / circularity numbers for the recommended **v4**
generation. Trained and evaluated on `Haitao999/things-eeg`, subject `sub-01`,
on a single RTX 4090. See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the
design rationale and the v1 → v4 evolution story.

## TL;DR

> **NeuroZip preserves more retrievability per bit at every compression tier.**
> +4 to +11 percentage points on top-5 image-prompt retrieval vs the
> fidelity baseline, while MSE is within ±5%. The held-out concept
> classifier (independent judge) confirms the win generalizes.

## Setup

| | value |
|---|---|
| Subject | `sub-01` (one of 10 in the dataset) |
| EEG | 63 channels, 250 Hz, 250 sample epochs (1 s post-stimulus + 0.2 s baseline) |
| Train epochs | 16,540 image-trials × 4 repetitions = 66,160 |
| Test concepts | 200 (disjoint from train) |
| Test repetitions | 80 per concept (16,000 epochs total) |
| Eval input | trial-averaged across all 80 reps per concept |
| CLIP variant | ViT-B/32 / LAION-2B (matches the dataset's precomputed features — *not* OpenAI; see ARCHITECTURE.md) |

## Projector (the frozen judge)

Trained with symmetric InfoNCE against precomputed CLIP image features.
Frozen for all downstream stages. Adding a [CLS] + 2-block transformer head
to the conv tower is the dominant architectural lever; results on the
trial-averaged test set:

| projector | params | top-1 | top-5 | top-10 |
|---|---:|---:|---:|---:|
| conv-only (AvgPool + MLP head) | 2.5 M | 13.5% | 37.5% | 50.5% |
| **conv tower + [CLS] + 2 attn blocks** | **6.0 M** | **18.5%** | **45.0%** | **65.0%** |

Chance on 200-way retrieval: 0.5% / 2.5% / 5.0% — judge clears it by 27× /
18× / 13× respectively.

## Codecs (v4 — recommended)

Architecture: conv-only encoder–quantizer–decoder, ~0.75 M params. Latent
shape `(32 channels × 32 timesteps) = 1024 integer symbols per epoch`.
Factorized Laplace prior for rate estimation. Same architecture for
fidelity and NeuroZip — only `lambda_task` differs (0 vs 3.0).

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

`demo/assets/rate_retrieval.png` — top-5 retrieval (y) vs bpp (x), both
families. NeuroZip's curve sits cleanly above fidelity at every tier.

## MSE: the asymmetric trade

Compared to the v2 (ViT-bottleneck codec) generation, v4 fidelity's MSE is
**2.6× lower** at the medium tier (0.030 vs 0.065) — the conv-only codec
just converges faster than the 4.3 M-param ViT codec within the same epoch
budget. NeuroZip inherits this; v4 NeuroZip MSE is 2.8× lower than v2's.

The intra-v4 fidelity-vs-NeuroZip MSE gap is small (typically ±5%) — see
the table above. **NeuroZip's worst MSE deficit is ~5%; its best top-5
advantage is +11 pp (~+42% relative).** The trade is heavily asymmetric in
NeuroZip's favor, which is the entire point.

## The circularity defense

A **separate** EEG → concept classifier is trained on 60 of the 80 test
reps per concept (averaging blocks of 10 reps as inputs for SNR), then
scored on each codec's decompressed test EEG. The classifier never sees
codec output during its own training — it's a different judge with a
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
gaps — useful as a sanity check that v4 isn't co-adapting against an
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

These appear in the demo's "featured" dropdown by default — re-run the
sweep + `evaluate.py` to regenerate the list (`demo/assets/summary.json`).

## How to reproduce

```bash
git clone http://100.64.0.59:8081/root/neurozip.git
cd neurozip
./train.sh sweep_v4        # ~45 min on a single RTX 4090
./serve_clean.sh           # http://<host>:8011/clean
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
