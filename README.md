# NeuroZip

**Task-aware neural compression for brain/biomedical signals — ffmpeg, but it keeps the *meaning*.**

A normal codec optimizes raw fidelity (does the reconstruction look/measure like the original?).
NeuroZip optimizes **decodable semantic content**: a frozen judge model sits in the compressor's
training loop, so the codec learns to throw away bits that don't matter to downstream analysis and
keep the ones that do. The result is a signal you can compress hard while it stays
**text-searchable** and **retrievable** — where a fidelity-only codec at the same bitrate loses it.

> **One-liner:** the same frozen model that decides what to throw away during compression is what
> lets you text-search what you kept.

First target: **EEG** (THINGS-EEG), with a CLIP-based judge so you can type a word and retrieve the
EEG epochs recorded while a subject looked at that thing — even after aggressive compression.

> **Status: v0.1.0-beta.1.** The CLI, the local drag-and-drop UI, and the importable
> frozen layer all work today. `compress` / `decompress` / `search` currently emit
> **placeholder (randomized)** results — the real neural codec is under construction.

## Install

**Homebrew** (recommended):
```bash
brew install xerneas3318/tap/neurozip
```

**curl**:
```bash
curl -fsSL https://raw.githubusercontent.com/xerneas3318/NeuroZip/main/install.sh | bash
```

**pip** (from source):
```bash
pip install "neurozip @ git+https://github.com/xerneas3318/NeuroZip"
# with the neural layer / training stack:
pip install "neurozip[ml] @ git+https://github.com/xerneas3318/NeuroZip"
```

The core install is **stdlib-only** (fast); PyTorch and friends live behind the `ml` extra.

## CLI (ffmpeg-style)

```bash
neurozip compress   path/to/folder/   -o out.nz   --ratio 50
neurozip decompress out.nz            -o restored/
neurozip search     "accordion"       --in out.nz --topk 5
neurozip serve                                          # local drag-and-drop UI (127.0.0.1:7878)
neurozip ui                                             # alias for serve
```

## Use it as a frozen layer (no embedding to train)

Drop NeuroZip in where you'd otherwise put a trainable embedding. It's a pretrained,
**frozen** encoder, so you never spend gradients updating it — you train only the rest
of your model, gradients still flow through:

```python
import torch
from neurozip import NeuroZipLayer

embed = NeuroZipLayer.from_pretrained("eeg-clip-b32")   # frozen, requires_grad=False
x = torch.randn(8, 63, 200)        # (batch, channels, time)
z = embed(x)                       # (8, 512) semantic embedding — no grad spent here
logits = my_head(z)                # train only my_head
```

> Requires `pip install "neurozip[ml]"`. Beta: weights are frozen-random until a
> checkpoint is published; the API is stable.

## How it works

1. **Frozen judge** — a pretrained CLIP plus a small EEG→CLIP projector (trained once, then
   frozen) that maps an EEG epoch into CLIP's shared image/text space.
2. **Codec** — a 1D-conv autoencoder + learned entropy model giving a real bitrate.
3. **Task-aware loss** — `rate + λ_recon·MSE + λ_task·(1 − cos(judge(decoded), CLIP_image(seen)))`.
   Gradients flow *through* the frozen judge into the codec, so compression is optimized for
   retrievability, not pixel/sample fidelity.

The headline result is a **rate vs retained-retrieval** curve: NeuroZip stays flat as you compress
while a fidelity-only codec falls off a cliff.

## Layout

| Path | Role |
|------|------|
| `neurozip/data.py` | dataset loading / normalization |
| `neurozip/clip_proj.py` | frozen CLIP + EEG→CLIP projector (the judge) |
| `neurozip/codec.py` | autoencoder codec + task-aware loss |
| `neurozip/train.py` | training (stage / λ / ratio flags) |
| `neurozip/evaluate.py` | metrics + demo-asset generation |
| `neurozip/cli.py` | ffmpeg-like command line |
| `ui/demo.html` | self-contained demo page |

## License

TBD.
