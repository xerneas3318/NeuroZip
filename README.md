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

> Status: early / hackathon scaffold. APIs and modules are stubs under active construction.

## Install (dev)

```bash
git clone https://github.com/xerneas3318/NeuroZip
cd NeuroZip
pip install -e .
```

Eventually: `brew install xerneas3318/tap/neurozip`.

## CLI (planned, ffmpeg-style)

```bash
neurozip compress   path/to/eeg/      -o out.nz   --ratio 50
neurozip decompress out.nz            -o restored/
neurozip search     "accordion"       --in out.nz --topk 5
neurozip ui                                              # drag-and-drop interface
```

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
