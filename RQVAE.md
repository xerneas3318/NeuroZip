# RQ-VAE latent (residual vector quantization)

Branch off `rian`. Replaces the codec's **latent space** — scalar-quantize +
factorized-Laplace → **Residual Vector Quantization** (SoundStream / EnCodec
style). The RVQ is implemented correctly; this doc reports what it actually buys,
which is **not** what the first version of this file claimed.

## What changed

`rqvae.py`:
- `VectorQuantizerEMA` — learned codebook, EMA updates, **+ dead-code restart**
  (revives unused entries from the live batch). Without the restart the codebook
  collapsed to **6 of 1024** codes, which made the nominal `D·log2(K)` rate a
  lie; with it, all 1024 are used.
- `ResidualVQ` — `D` codebooks; each quantizes the residual of the previous ones.
- `EEGCodecRQ` — reuses rian's `EEGEncoder`/`EEGDecoder`; swaps the scalar
  quantizer + Laplace prior for `ResidualVQ`. Same reporting interface.

Rate is exact and fixed: `bits/epoch = 32 tokens · D · log2(K)`, independent of
`c_lat`. So representational capacity is decoupled from bitrate, and you get a
discrete code stream (useful if the project later wants EEG tokens / AR modeling).

## The honest result

The original claim here — *"RQ-VAE = 3.5–4× lower MSE than v4, an upgrade"* — was
**a training-protocol artifact, not a property of residual VQ.** Two controls
prove it.

**Control 1 — same protocol, different quantizer.** Train scalar and RVQ both on
the averaged + scale-aug protocol. They tie:

| codec | quantizer | MSE ↓ | var-exp | ratio |
|---|---|---:|---:|---:|
| scalar-attn (`codec_fidelity_72x`) | scalar + Laplace | **0.0070** | 80% | 67× |
| RQ-VAE (`codec_rqvae_72x`) | residual-VQ, D=11×K=1024 | 0.0074 | 79% | 72× |

Scalar is *marginally better*. RVQ is not an MSE upgrade. (Bypassing the RVQ
quantizer entirely only moves MSE 0.0074→0.0065 — the encoder/decoder does the
work, the quantizer family barely matters.)

**Control 2 — same v4 architecture, different protocol.** Take rian's exact
conv-only v4 codec and only swap its training protocol. It recovers most of the
"gap" with no codec change at all:

| v4 architecture (conv-only, c_lat=32) | protocol | MSE ↓ | var-exp | ratio |
|---|---|---:|---:|---:|
| v4 (rian) | single-trial | 0.0317 | 10% | 64× |
| v4-arch | averaged + scale-aug | **0.0098** | 72% | 108× |

The whole "4×" effect is the protocol. v4 trains on single trials (magnitude
≈1.0) but is scored on 80-rep trial-averages (magnitude ≈0.19), so it
**over-shoots amplitude** — its reconstruction std is 0.287 vs the target's
0.188. Fix the protocol and the same conv-only codec lands at 0.0098 at *higher*
(108×) compression than RVQ.

## The metric that actually matters: retrieval

NeuroZip's thesis is **task-aware retrieval**, not waveform MSE (rian's
`results.md`: *"worst MSE deficit ~5%, best top-5 advantage +11pp — the trade is
asymmetric"*). The MSE-only benchmark never measured it. `benchmark_retrieval.py`
adds it — top-k through the frozen projector `P`, vs per-concept CLIP image
features:

| codec | ratio | MSE | top1 | **top5** | top10 |
|---|---:|---:|---:|---:|---:|
| raw EEG (uncompressed) | 1× | — | 14.5 | 45.5 | 62.5 |
| RQ-VAE | 72× | 0.0074 | 16.0 | **40.5** | 60.0 |
| scalar-attn | 67× | 0.0070 | 15.0 | 40.0 | 57.5 |
| v4-conv (avg protocol) | 108× | 0.0098 | 14.5 | 39.0 | 55.0 |
| v4 (single-trial) | 64× | 0.0317 | 8.0 | 25.0 | 38.5 |

RQ-VAE (40.5) ≈ scalar (40.0) on top-5 — tied within single-seed noise.
Everything below them is the protocol gap again.

## Correctness checks (the RVQ is implemented right)

- Codes are discrete `int64` in `[0, 1023]`; all 11 codebooks well-used (460–944
  unique of 1024 after the dead-code fix).
- Quantizer is a genuine, active bottleneck (MSE with quant 0.0074 > bypass 0.0065).
- **Residual energy decays monotonically across the 11 stages:**
  `0.118 → 0.046 → 0.036 → 0.030 → 0.024 → 0.020 → 0.016 → 0.013 → 0.011 → 0.009 → 0.007 → 0.006`
  — the defining signature of correct residual quantization.

## Bottom line

RVQ here is a **clean fixed-rate discrete latent**, correctly implemented — but
at this scale it is **MSE-neutral and retrieval-neutral** vs the scalar codec.
It's worth keeping for the discrete-token / AR direction, not as a fidelity or
retrieval "upgrade." The dramatic numbers vs v4 were the training protocol, which
applies equally to the scalar codec.

## Run

```bash
# RVQ codec (D controls the tier)
python train_rqvae.py --num-quantizers 11 --codebook-size 1024 --epochs 60 \
                      --out checkpoints/codec_rqvae_72x.pt
# matched-protocol baselines (the fair comparison)
python train_fidelity.py --c-lat 96 --hidden 160 --n-attn 2 --epochs 60 \
                         --out checkpoints/codec_fidelity_72x.pt
python train_fidelity.py --c-lat 32 --hidden 128 --n-attn 0 --lambda-rate 0.03 \
                         --epochs 60 --out checkpoints/codec_v4arch_avg.pt
# the metric that matters
python benchmark_retrieval.py --models \
   RQ-VAE=checkpoints/codec_rqvae_72x.pt \
   scalar-attn=checkpoints/codec_fidelity_72x.pt \
   v4-conv-avg=checkpoints/codec_v4arch_avg.pt
# visualization (unchanged)
./serve_rqvae.sh        # viewer on http://localhost:8011/
```
