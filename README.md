# NeuroZip

**Task-aware neural compression for brain signals - ffmpeg, but it keeps the *meaning*.**

A normal codec optimizes raw fidelity. NeuroZip optimizes **decodable semantic content**: a
frozen CLIP-space judge sits in the codec's training loop, so it learns to spend bits on the
parts of an EEG epoch that stay *retrievable* and drop the rest. At a matched bitrate, the
task-aware codec beats a fidelity-only codec on text/image retrieval from the compressed signal.

> **One-liner:** the same frozen model that decides what to throw away during compression is
> what lets you text-search what you kept.

Trained on THINGS-EEG (63 channels x 250 samples, ViT-B-32 CLIP space). Measured, at the `high`
tier: **neurozip image top-1 = 0.14 vs fidelity 0.08** at ~the same bitrate.

> Status: **v0.2.0-beta.1** - real trained models, real CLI, real upload UI.

## Install

```bash
brew install xerneas3318/tap/neurozip      # the CLI + UI (stdlib-only core)
pip install "neurozip[ml]"                 # add the torch inference stack
neurozip download                          # fetch the trained models -> ~/.neurozip
```

`brew`/core install is dependency-free. Real inference (compress, embed, the layer, the UI)
needs the `ml` extra (torch) and the model bundle from `neurozip download`.

## CLI

```bash
neurozip info                              # list tiers + measured scores
neurozip sample --idx 2 -o cat.npy         # write a demo EEG epoch
neurozip compress cat.npy -o cat.nz --tier high          # real codec: bpp, ratio, latents
neurozip compress cat.npy --variant fidelity             # the fidelity-only baseline
neurozip decompress cat.nz -o cat.recon.npy
neurozip embed cat.npy                     # 512-d CLIP-space embedding (the frozen judge)
neurozip serve                             # local upload UI at 127.0.0.1:7878
```

Tiers: `low | med | high | xhigh` (more bits -> better fidelity + retrieval). Variants:
`neurozip` (task-aware) and `fidelity` (matched-bitrate baseline for comparison).

## Upload UI

`neurozip serve` opens a page where you drag in an EEG epoch `.npy` (or pick a bundled demo
epoch), choose a tier/variant, and get the real bitrate, compression ratio, reconstruction MSE,
a raw-vs-decoded heatmap, and the neurozip-vs-fidelity retrieval table.

## Use it as a frozen layer (no embedding to train)

Drop NeuroZip in where you'd put a trainable embedding. It's the trained, frozen EEG->CLIP
projector, so you never spend gradients on it - you train only the rest of your model:

```python
import torch
from neurozip import NeuroZipLayer

embed = NeuroZipLayer.from_pretrained()    # frozen, requires_grad=False
x = torch.randn(8, 63, 250)                # (batch, channels, time), normalized EEG
z = embed(x)                               # (8, 512) unit-norm - no grad spent here
logits = my_head(z)                        # train only my_head
```

## How it works

1. **Frozen judge** - an EEG->CLIP projector (conv tower + transformer head) trained to map an
   epoch into CLIP's image/text space, then frozen.
2. **Codec** - a 1D-conv autoencoder + factorized-Laplace entropy model giving a real bitrate.
3. **Task-aware loss** - `rate + lambda_recon*MSE + lambda_task*(1 - cos(judge(decoded), CLIP_img(seen)))`.
   Gradients flow *through* the frozen judge into the codec, so compression is optimized for
   retrievability, not sample fidelity.

See `ARCHITECTURE.md` for the full design and the three easy-to-get-wrong spots (quantizer,
gradient flow through the frozen judge, normalization stats).

## License

[PolyForm Noncommercial License 1.0.0](LICENSE): public source, free for any **noncommercial**
use (personal, research, education, nonprofit). **Selling it is not permitted.** Source-available,
not an OSI "open source" license (which cannot forbid selling).
