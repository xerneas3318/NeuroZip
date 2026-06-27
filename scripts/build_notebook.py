"""Emit notebook.ipynb — standalone NeuroZip viewer.

Re-run this script to regenerate the notebook after editing cell contents
here. Keeps the cell source readable as Python rather than JSON-escaped
strings.
"""

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebook.ipynb"

cells = []

def _id():
    return uuid.uuid4().hex[:8]

def md(src: str):
    cells.append({"cell_type": "markdown", "id": _id(), "metadata": {}, "source": src})

def code(src: str):
    cells.append({"cell_type": "code", "id": _id(), "metadata": {},
                  "source": src, "outputs": [], "execution_count": None})


md("""# NeuroZip viewer

A standalone alternative to the Flask demo (`serve.py`). Load checkpoints +
test data once, then explore retrievals, reconstructions, latents, and
test-set aggregates inline. Edit any cell's input and re-run.

**Prereqs:** finished training (`./train.sh` or `./train.sh v4`), so
`checkpoints/clip_proj.pt`, codec checkpoints, and `data/` are all on disk.

**Suggested run order:** top to bottom, then iterate on the inputs in
sections 3 / 4 / 5.
""")


md("## 0. Setup")

code("""# Standard imports + paths.
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Make the project root importable so we can reuse data.py / clip_proj.py / codec.py.
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from data import ThingsEEG, N_CHANNELS, N_TIMES, IMG_DIR
from clip_proj import EEGProjector, load_frozen_projector, retrieval_topk
from codec import EEGCodec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINTS = ROOT / "checkpoints"
print(f"device: {DEVICE}  |  ROOT: {ROOT}")
""")


code("""# Pick a codec generation. Auto-detect: prefer v4 (conv-only codec +
# attention judge, the cleanest NeuroZip story) if its checkpoints exist,
# else fall back to v2 (ViT codecs).
PROJECTOR_PATH = CHECKPOINTS / "clip_proj.pt"
CLASSIFIER_PATH = CHECKPOINTS / "holdout_classifier.pt"
SCORES_PATH = CHECKPOINTS / "scores.json"

def _ckpts(prefix):
    return sorted(p.stem for p in CHECKPOINTS.glob(f"*{prefix}*.pt"))

CODEC_PREFIX = None
for cand in ("v4", "v2", "v3", "v1"):
    files = _ckpts(cand)
    if any(n.startswith("fidelity") for n in files) and any(n.startswith("neurozip") for n in files):
        CODEC_PREFIX = cand; break
if CODEC_PREFIX is None:
    raise SystemExit("No codec checkpoints found. Run ./train.sh first.")

available = _ckpts(CODEC_PREFIX)
print(f"using codec generation: {CODEC_PREFIX}")
for n in available: print(f"  {n}")
""")


code("""# Load projector + codecs + classifier + dataset.
proj = load_frozen_projector(PROJECTOR_PATH, DEVICE)
print(f"projector: {sum(p.numel() for p in proj.parameters())/1e6:.2f}M params, "
      f"n_attn={getattr(proj, 'n_attn', 0)}")

def load_codec(name: str) -> EEGCodec:
    state = torch.load(CHECKPOINTS / f"{name}.pt", weights_only=False, map_location=DEVICE)
    cfg = state.get("config", {})
    m = EEGCodec(c_lat=cfg.get("c_lat", 32), hidden=cfg.get("hidden", 128),
                  n_attn=cfg.get("n_attn", 0)).to(DEVICE)
    m.load_state_dict(state["model"]); m.eval()
    return m

codecs = {n: load_codec(n) for n in available}
print(f"loaded {len(codecs)} codecs")

# Pair them up by tier.
fid_names = sorted([n for n in codecs if n.startswith("fidelity")])
nzn_names = sorted([n for n in codecs if n.startswith("neurozip")])
print(f"fidelity tiers: {fid_names}")
print(f"neurozip tiers: {nzn_names}")

# Held-out classifier (optional).
clf = None
if CLASSIFIER_PATH.exists():
    from train import HoldoutClassifier
    st = torch.load(CLASSIFIER_PATH, weights_only=False, map_location=DEVICE)
    clf = HoldoutClassifier(n_classes=st["n_classes"], hidden=st["hidden"],
                              n_attn=st.get("n_attn", 0), attn_heads=st.get("attn_heads", 4)).to(DEVICE)
    clf.load_state_dict(st["model"]); clf.eval()
    print("loaded held-out classifier")

# Dataset (test split; trial-averaged for evaluation).
test = ThingsEEG(split="test")
eeg_avg, concept_list_avg, img_paths = test.trial_averaged()
text_features, concept_list = test.concept_clip_text()
img_features = test.concept_clip_img()
print(f"test set: {len(concept_list)} concepts, eeg shape {tuple(eeg_avg.shape)}")
""")


md("## 1. Per-codec metrics table")


code("""# Read evaluate.py's scores.json if present, else compute fresh.
if SCORES_PATH.exists():
    scores = json.loads(SCORES_PATH.read_text())
    rows = [(n, scores[n]) for n in sorted(scores) if n in codecs]
else:
    # Cheap inline eval if scores.json is missing.
    rows = []
    with torch.no_grad():
        x = eeg_avg.to(DEVICE)
        gallery_img = img_features.to(DEVICE)
        gold = torch.arange(x.size(0), device=DEVICE)
        for n, c in codecs.items():
            xh, bits = c.compress_then_reconstruct(x)
            embs = proj(xh)
            tk = retrieval_topk(embs, gallery_img, gold)
            rows.append((n, {"bpp": c.bpp_floor(bits.item()),
                             "compression_ratio_vs_fp16": 16.0/c.bpp_floor(bits.item()),
                             "mse": F.mse_loss(xh, x).item(),
                             "retrieval_image": tk,
                             "holdout_top1": None}))

# Pretty-print.
hdr = f"{'codec':24s} {'bpp':>6s} {'ratio':>6s} {'mse':>7s} {'top1':>6s} {'top5':>6s} {'top10':>6s} {'hold':>6s}"
print(hdr); print("-"*len(hdr))
for n, m in rows:
    h = m.get("holdout_top1")
    print(f"{n:24s} {m['bpp']:6.3f} {m['compression_ratio_vs_fp16']:5.0f}× "
          f"{m['mse']:7.4f} "
          f"{m['retrieval_image']['top1']*100:5.1f}% "
          f"{m['retrieval_image']['top5']*100:5.1f}% "
          f"{m['retrieval_image']['top10']*100:5.1f}% "
          f"{'n/a' if h is None else f'{h*100:5.1f}%'}")
""")


md("## 2. Rate–retrieval plot")


code("""# Plot top-5 retrieval vs bpp for both families. The "money plot".
fid_rows = sorted([(m['bpp'], m['retrieval_image']['top5']) for n,m in rows if n.startswith('fidelity')])
nzn_rows = sorted([(m['bpp'], m['retrieval_image']['top5']) for n,m in rows if n.startswith('neurozip')])

fig, ax = plt.subplots(figsize=(8, 4.5))
if fid_rows:
    xs, ys = zip(*fid_rows); ax.plot(xs, [100*y for y in ys], "o--", color="#7c8497", label="fidelity-only codec")
if nzn_rows:
    xs, ys = zip(*nzn_rows); ax.plot(xs, [100*y for y in ys], "o-", color="#d62728", label="NeuroZip (task-aware)")
ax.set_xlabel("bits per sample (lower = more compressed)")
ax.set_ylabel("retrieval top-5 (%)  [image prompt -> EEG]")
ax.set_title("NeuroZip vs. fidelity codec @ matched bpp")
ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
plt.show()
""")


md("""## 3. Reconstruction viewer

Edit `EPOCH_IDX` below and re-run to see how each codec reconstructs that
specific epoch. The ground-truth concept's stimulus image is shown alongside.""")


code("""EPOCH_IDX = 150     # 0..199; try 150 (robot), 126 (ostrich), 190 (unicycle)
# Pick the codecs to compare. Default: lowest-bpp neurozip + matching fidelity.
M_NZN = nzn_names[0] if nzn_names else None
M_FID = fid_names[0] if fid_names else None
print(f"epoch #{EPOCH_IDX}  concept = {concept_list[EPOCH_IDX]!r}")
print(f"models: NeuroZip = {M_NZN!r}   fidelity = {M_FID!r}")

# Show the stimulus image.
img_path = IMG_DIR / img_paths[EPOCH_IDX]
if img_path.exists():
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(Image.open(img_path)); ax.axis('off')
    ax.set_title(f"stimulus: {concept_list[EPOCH_IDX]}", fontsize=10)
    plt.show()
else:
    print(f"(no image at {img_path})")
""")


code("""# 3-row heatmap: raw vs each model's reconstruction (shared color scale).
@torch.no_grad()
def reconstruct(epoch_idx, name):
    x = eeg_avg[epoch_idx:epoch_idx+1].to(DEVICE)
    xh, bits = codecs[name].compress_then_reconstruct(x)
    return x.squeeze(0).cpu(), xh.squeeze(0).cpu(), bits.item()

raw, hat_nzn, bits_nzn = reconstruct(EPOCH_IDX, M_NZN)
_,   hat_fid, bits_fid = reconstruct(EPOCH_IDX, M_FID)
panels = [
    ("raw (trial-averaged 80 reps)", raw.numpy()),
    (f"{M_NZN} (bpp={codecs[M_NZN].bpp_floor(bits_nzn):.3f}, mse={(raw-hat_nzn).pow(2).mean():.4f})", hat_nzn.numpy()),
    (f"{M_FID} (bpp={codecs[M_FID].bpp_floor(bits_fid):.3f}, mse={(raw-hat_fid).pow(2).mean():.4f})", hat_fid.numpy()),
]
vmax = float(np.max(np.abs([p[1] for p in panels])))
fig, axes = plt.subplots(3, 1, figsize=(9, 5), sharex=True, gridspec_kw={"hspace":0.35})
for ax, (t, arr) in zip(axes, panels):
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[-200, 996, 63, 0])
    ax.set_title(t, fontsize=10, loc="left"); ax.set_ylabel("channel")
axes[-1].set_xlabel("time after stimulus (ms)")
fig.colorbar(im, ax=axes, shrink=0.7, label="normalized amplitude (σ)")
plt.show()
""")


code("""# Per-channel waveform overlay (6 representative scalp channels).
CHANNELS = [4, 18, 25, 32, 47, 60]
t_ms = np.linspace(-200, 996, N_TIMES)

fig, axes = plt.subplots(len(CHANNELS), 1, figsize=(9, 1.1*len(CHANNELS)+0.4), sharex=True)
for ax, ch in zip(axes, CHANNELS):
    ax.plot(t_ms, raw[ch].numpy(), color="black", lw=1.2, label="raw")
    ax.plot(t_ms, hat_nzn[ch].numpy(), color="#d62728", lw=1.0, alpha=0.85, label="NeuroZip")
    ax.plot(t_ms, hat_fid[ch].numpy(), color="#7c8497", lw=1.0, alpha=0.85, ls="--", label="fidelity")
    ax.axvline(0, color="#bbb", lw=0.7); ax.set_ylabel(f"ch {ch}")
axes[0].legend(loc="upper right", fontsize=8); axes[-1].set_xlabel("time (ms)")
fig.suptitle(f"per-channel waveforms — epoch #{EPOCH_IDX} ({concept_list[EPOCH_IDX]})", fontsize=11)
fig.tight_layout(); plt.show()
""")


code("""# Quantized latent (the "shipped" data) for both codecs.
@torch.no_grad()
def latent(epoch_idx, name):
    x = eeg_avg[epoch_idx:epoch_idx+1].to(DEVICE)
    y = codecs[name].encoder(x)
    return torch.round(y).cpu().squeeze(0).numpy()

z_nzn, z_fid = latent(EPOCH_IDX, M_NZN), latent(EPOCH_IDX, M_FID)
vmax = float(max(np.abs(z_nzn).max(), np.abs(z_fid).max()))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, z, name in zip(axes, [z_nzn, z_fid], [M_NZN, M_FID]):
    im = ax.imshow(z, aspect="auto", cmap="PuOr", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{name}  (max |z|={int(np.abs(z).max())}, unique={len(np.unique(z))})", fontsize=10)
    ax.set_xlabel("latent time"); ax.set_ylabel("latent channel")
fig.colorbar(im, ax=axes, shrink=0.7, label="integer symbol")
plt.show()
""")


md("""## 4. Free-text retrieval

Encode any phrase via CLIP, then retrieve the EEG epoch whose decompressed
projection is closest in CLIP space. Compare NeuroZip vs fidelity at the
chosen tier.""")


code("""# Lazy-load CLIP text encoder (LAION-2B variant, matches the dataset features).
import open_clip
clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
clip_model = clip_model.eval().to(DEVICE)
clip_tok = open_clip.get_tokenizer("ViT-B-32")
for p in clip_model.parameters(): p.requires_grad_(False)
print(f"CLIP loaded ({sum(p.numel() for p in clip_model.parameters())/1e6:.0f}M params)")

# Precompute decompressed-EEG embeddings for every codec (cached).
@torch.no_grad()
def codec_embeddings(name):
    x = eeg_avg.to(DEVICE)
    xh, _ = codecs[name].compress_then_reconstruct(x)
    return F.normalize(proj(xh), dim=-1).cpu()

eeg_emb_cache = {n: codec_embeddings(n) for n in codecs}
print(f"cached {len(eeg_emb_cache)} codec embedding sets")
""")


code("""QUERY = "a small flightless bird"      # edit me
TIER  = 0                                # 0 = highest compression, last = lowest
K     = 5

m_nzn = nzn_names[TIER]; m_fid = fid_names[TIER]
print(f"query: {QUERY!r}   |   tier: {m_nzn} vs {m_fid}")

with torch.no_grad():
    q = F.normalize(clip_model.encode_text(clip_tok([QUERY]).to(DEVICE)), dim=-1).cpu().squeeze(0)

# Find the closest of the 200 known concepts (for "gold" marking).
sim_to_text = F.normalize(text_features, dim=-1) @ q
gold_idx = int(sim_to_text.argmax()); gold_concept = concept_list[gold_idx]
print(f"nearest known concept: {gold_concept!r}  (cos={sim_to_text[gold_idx]:.3f})")

# Retrieve from each codec.
def topk_for(name):
    sims = eeg_emb_cache[name] @ q
    top = sims.topk(K)
    return top.indices.tolist(), top.values.tolist()

nzn_idx, nzn_sc = topk_for(m_nzn)
fid_idx, fid_sc = topk_for(m_fid)
print(f"\\nNeuroZip top-{K}: {[concept_list[i] for i in nzn_idx]}")
print(f"fidelity top-{K}: {[concept_list[i] for i in fid_idx]}")

# Show retrieved images side by side.
fig, axes = plt.subplots(2, K+1, figsize=(2.2*(K+1), 4.6))
for row, (idx, sc, title) in enumerate([(nzn_idx, nzn_sc, f"NeuroZip ({m_nzn})"),
                                         (fid_idx, fid_sc, f"fidelity ({m_fid})")]):
    # leftmost: gold stimulus
    p = IMG_DIR / img_paths[gold_idx]
    axes[row, 0].imshow(Image.open(p) if p.exists() else np.zeros((100,100,3)))
    axes[row, 0].set_title(f"gold: {gold_concept}", fontsize=9, color="#4caf50")
    axes[row, 0].axis('off')
    for c, (i, s) in enumerate(zip(idx, sc), start=1):
        ip = IMG_DIR / img_paths[i]
        axes[row, c].imshow(Image.open(ip) if ip.exists() else np.zeros((100,100,3)))
        color = "#4caf50" if i == gold_idx else "#888"
        axes[row, c].set_title(f"#{c} {concept_list[i]}\\n(sim {s:.2f})", fontsize=8, color=color)
        axes[row, c].axis('off')
    axes[row, 0].annotate(title, xy=(-0.15, 0.5), xycoords='axes fraction',
                          rotation=90, va='center', ha='center', fontsize=10)
fig.tight_layout()
plt.show()
""")


md("## 5. Test-set aggregates")


code("""# Per-channel reconstruction MSE.
@torch.no_grad()
def per_channel_mse(name, batch=200):
    x = eeg_avg.to(DEVICE)
    out = []
    for i in range(0, x.size(0), batch):
        xh, _ = codecs[name].compress_then_reconstruct(x[i:i+batch])
        out.append((x[i:i+batch] - xh).pow(2).mean(dim=(0,2)).cpu())
    return torch.stack(out).mean(0).numpy()

per_nzn = per_channel_mse(M_NZN); per_fid = per_channel_mse(M_FID)
fig, ax = plt.subplots(figsize=(9, 3))
xs = np.arange(N_CHANNELS); w = 0.4
ax.bar(xs - w/2, per_nzn, w, color="#d62728", alpha=0.85, label=f"NeuroZip ({M_NZN})")
ax.bar(xs + w/2, per_fid, w, color="#7c8497", alpha=0.85, label=f"fidelity ({M_FID})")
ax.set_xlabel("channel index"); ax.set_ylabel("mean squared error")
ax.set_title("per-channel reconstruction error"); ax.legend()
plt.show()
""")


code("""# Entropy distribution (bits per latent symbol) across the test set.
@torch.no_grad()
def bits_per_symbol(name):
    x = eeg_avg.to(DEVICE)
    y = codecs[name].encoder(x)
    y_int = torch.round(y)
    upper = codecs[name].prior._cdf(y_int + 0.5)
    lower = codecs[name].prior._cdf(y_int - 0.5)
    p = (upper - lower).clamp_min(1e-9)
    return (-torch.log2(p)).flatten().cpu().numpy()

bits_nzn = bits_per_symbol(M_NZN); bits_fid = bits_per_symbol(M_FID)
fig, ax = plt.subplots(figsize=(9, 3))
ax.hist(bits_nzn, bins=60, density=True, alpha=0.55, color="#d62728", label=f"NeuroZip ({M_NZN})")
ax.hist(bits_fid, bins=60, density=True, alpha=0.55, color="#7c8497", label=f"fidelity ({M_FID})")
ax.set_xlabel("bits per latent symbol"); ax.set_ylabel("density")
ax.set_title("entropy distribution across all latent positions"); ax.legend()
plt.show()
""")


# Assemble notebook JSON
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(cells)} cells)")
