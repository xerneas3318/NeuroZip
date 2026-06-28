"""NeuroZip live-inference Flask backend.

Endpoints (all relative to http://host:port):

  GET  /                        -> demo.html
  GET  /demo/assets/...         -> static demo assets (images, plot, summary.json)
  GET  /api/health              -> {"ok": true}
  GET  /api/models              -> {name: {bpp, ratio, ...}}
  GET  /api/concepts            -> ["aircraft_carrier", ...]
  POST /api/retrieve            -> body {text|concept_idx, model_a, model_b, k}
                                   -> {model_name: {top_k: [{idx, concept, sim}], gold_in_topk}}
  POST /api/reconstruct         -> body {epoch_idx, models: [...]} -> PNG figs (base64)
        kinds=heatmap|diff|waveforms|latent|channel_mse

CLIP model: ViT-B-32 with `laion2b_s34b_b79k` weights, matching the
precomputed image+text features shipped with the Haitao999/things-eeg
dataset. Verified by image-image cosine ~0.98 against the dataset's
precomputed features.
"""

from __future__ import annotations
import argparse, base64, io, json, os, time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory, send_file, abort

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import ThingsEEG, N_CHANNELS, N_TIMES, CLIP_DIM, IMG_DIR
from clip_proj import load_frozen_projector
from codec import EEGCodec, CHECKPOINTS
from train import HoldoutClassifier

ROOT = Path(__file__).parent
DEMO_DIR = ROOT / "demo"
ASSETS_DIR = DEMO_DIR / "assets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__, static_folder=None)


# ============================================================================
# Lazy global state (loaded once on first request, cached for the process)
# ============================================================================

class State:
    proj = None
    codecs: dict[str, EEGCodec] = {}
    classifier = None
    test_ds: ThingsEEG = None
    eeg_avg: torch.Tensor = None          # (200, 63, 250) trial-averaged test EEG
    concept_list: list[str] = None
    img_features: torch.Tensor = None      # (200, 512) CLIP image emb per concept
    text_features: torch.Tensor = None     # (200, 512) CLIP text emb per concept
    concept_img_paths: list[str] = None
    clip_model = None                      # open_clip ViT-B-32 LAION-2B
    clip_tok = None
    norm_mean: torch.Tensor = None
    norm_std: torch.Tensor = None
    eeg_emb_cache: dict[str, torch.Tensor] = {}   # model_name -> (200, 512) judge embeddings


def load_state():
    if State.test_ds is not None:
        return
    print("[serve] loading state...")
    t0 = time.time()

    # Test EEG + galleries
    test = ThingsEEG(split="test")
    State.test_ds = test
    eeg_avg, _texts, imgs = test.trial_averaged()
    State.eeg_avg = eeg_avg
    State.img_features = test.concept_clip_img()
    State.text_features, State.concept_list = test.concept_clip_text()
    State.concept_img_paths = imgs

    # Norm stats (for inverse-normalization on the viewer)
    nm = torch.load(ROOT / "data" / "norm_stats.pt", weights_only=False, map_location="cpu")
    State.norm_mean = nm["mean"]; State.norm_std = nm["std"]

    # Frozen judge
    State.proj = load_frozen_projector(CHECKPOINTS / "clip_proj.pt", DEVICE)

    # All codecs we can find
    for path in sorted(CHECKPOINTS.glob("*.pt")):
        name = path.stem
        if name in ("clip_proj", "holdout_classifier") or name.startswith("scores"):
            continue
        try:
            state = torch.load(path, weights_only=False, map_location=DEVICE)
            cfg = state.get("config", {})
            c = EEGCodec(c_lat=cfg.get("c_lat", 32),
                          hidden=cfg.get("hidden", 128),
                          n_attn=cfg.get("n_attn", 0)).to(DEVICE)
            c.load_state_dict(state["model"])
            c.eval()
            State.codecs[name] = c
        except Exception as e:
            print(f"  skipping {path.name}: {e}")

    # Independent classifier
    cpath = CHECKPOINTS / "holdout_classifier.pt"
    if cpath.exists():
        st = torch.load(cpath, weights_only=False, map_location=DEVICE)
        clf = HoldoutClassifier(n_classes=st["n_classes"], hidden=st["hidden"],
                                  n_attn=st.get("n_attn", 0),
                                  attn_heads=st.get("attn_heads", 4)).to(DEVICE)
        clf.load_state_dict(st["model"]); clf.eval()
        State.classifier = clf

    # CLIP text encoder. LAION-2B variant matches the precomputed features.
    import open_clip
    print("  loading CLIP (ViT-B-32 / laion2b)...")
    State.clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    State.clip_model.eval().to(DEVICE)
    for p in State.clip_model.parameters(): p.requires_grad_(False)
    State.clip_tok = open_clip.get_tokenizer("ViT-B-32")

    # Precompute decompressed-EEG -> judge embeddings for each codec.
    with torch.no_grad():
        x = eeg_avg.to(DEVICE)
        for name, c in State.codecs.items():
            xh, _ = c.compress_then_reconstruct(x)
            State.eeg_emb_cache[name] = State.proj(xh).cpu()

    print(f"[serve] state ready ({time.time()-t0:.1f}s)  "
          f"codecs={list(State.codecs)} clf={'yes' if State.classifier else 'no'}")


# ============================================================================
# Helpers
# ============================================================================

def fig_to_b64(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="#161922")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def encode_text(text: str) -> torch.Tensor:
    t = State.clip_tok([text]).to(DEVICE)
    e = State.clip_model.encode_text(t)
    return F.normalize(e, dim=-1).squeeze(0).cpu()


@torch.no_grad()
def reconstruct(epoch_idx: int, model_name: str):
    """Returns (raw_normalized, hat_normalized, bits_per_symbol) on CPU."""
    x = State.eeg_avg[epoch_idx:epoch_idx+1].to(DEVICE)
    c = State.codecs[model_name]
    xh, bits = c.compress_then_reconstruct(x)
    return x.squeeze(0).cpu(), xh.squeeze(0).cpu(), bits.item()


@torch.no_grad()
def latent(epoch_idx: int, model_name: str) -> torch.Tensor:
    x = State.eeg_avg[epoch_idx:epoch_idx+1].to(DEVICE)
    c = State.codecs[model_name]
    y = c.encoder(x)
    y_q = torch.round(y).cpu().squeeze(0)
    return y_q


def style_axes(ax):
    ax.set_facecolor("#0f1115")
    for s in ax.spines.values():
        s.set_color("#444a5e")
    ax.tick_params(colors="#d0d2d8", labelsize=8)
    ax.title.set_color("#e6e8ee")
    ax.xaxis.label.set_color("#bfc4d2"); ax.yaxis.label.set_color("#bfc4d2")


# ============================================================================
# Figure renderers
# ============================================================================

def fig_recon_heatmaps(epoch_idx: int, model_names: list[str]) -> str:
    """Raw vs reconstructions side-by-side. Vmin/vmax shared so colors compare."""
    raw, _, _ = reconstruct(epoch_idx, model_names[0])
    panels = [("raw EEG (trial-averaged 80 reps)", raw.numpy())]
    for m in model_names:
        _, xh, _ = reconstruct(epoch_idx, m)
        panels.append((f"{m} (decompressed)", xh.numpy()))
    vmax = float(np.max(np.abs([p[1] for p in panels])))
    vmin = -vmax
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.7 * n + 0.6), sharex=True,
                             gridspec_kw={"hspace": 0.45})
    if n == 1: axes = [axes]
    for ax, (title, arr) in zip(axes, panels):
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       extent=[-200, 996, 63, 0])
        ax.set_title(title, fontsize=11, loc="left", color="#e6e8ee", pad=4)
        ax.set_ylabel("channel")
        style_axes(ax)
    axes[-1].set_xlabel("time after stimulus (ms)")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02,
                      label="normalized amplitude (σ)")
    cb.ax.tick_params(colors="#d0d2d8")
    cb.set_label("normalized amplitude (σ)", color="#bfc4d2")
    fig.patch.set_facecolor("#161922")
    return fig_to_b64(fig)


def fig_recon_diff(epoch_idx: int, model_names: list[str]) -> str:
    """|raw - reconstruction| for each model, shared color scale."""
    raw, _, _ = reconstruct(epoch_idx, model_names[0])
    panels = []
    for m in model_names:
        _, xh, _ = reconstruct(epoch_idx, m)
        err = (raw - xh).abs().numpy()
        panels.append((f"|raw - {m}|  MSE={(raw-xh).pow(2).mean().item():.4f}", err))
    vmax = float(np.max([p[1].max() for p in panels]))
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.7*n+0.4), sharex=True,
                             gridspec_kw={"hspace": 0.45})
    if n==1: axes=[axes]
    for ax, (t, arr) in zip(axes, panels):
        im = ax.imshow(arr, aspect="auto", cmap="magma", vmin=0, vmax=vmax,
                       extent=[-200, 996, 63, 0])
        ax.set_title(t, fontsize=11, loc="left", color="#e6e8ee", pad=4)
        ax.set_ylabel("channel")
        style_axes(ax)
    axes[-1].set_xlabel("time after stimulus (ms)")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label="|error| (σ)")
    cb.ax.tick_params(colors="#d0d2d8")
    cb.set_label("|error| (σ)", color="#bfc4d2")
    fig.patch.set_facecolor("#161922")
    return fig_to_b64(fig)


def fig_channel_waveforms(epoch_idx: int, model_names: list[str],
                           channels: list[int] | None = None) -> str:
    if channels is None:
        channels = [4, 18, 25, 32, 47, 60]   # spread across the scalp
    raw, _, _ = reconstruct(epoch_idx, model_names[0])
    recons = {m: reconstruct(epoch_idx, m)[1] for m in model_names}
    ch_names = State.test_ds.eeg.shape  # just to verify
    t_ms = np.linspace(-200, 996, N_TIMES)
    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 1.05*n + 0.4), sharex=True)
    if n == 1: axes = [axes]
    color_raw = "#e6e8ee"
    nz_color = "#d62728"; fid_color = "#9aa0b2"
    for ax, ch in zip(axes, channels):
        ax.plot(t_ms, raw[ch].numpy(), color=color_raw, lw=1.2, label="raw")
        for m, xh in recons.items():
            c = nz_color if "neurozip" in m else fid_color
            ls = "-" if "neurozip" in m else "--"
            ax.plot(t_ms, xh[ch].numpy(), color=c, lw=1.0, alpha=0.85, ls=ls, label=m)
        ax.axvline(0, color="#444a5e", lw=0.7)
        ax.set_ylabel(f"ch {ch}", color="#bfc4d2", fontsize=9)
        style_axes(ax)
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.4, facecolor="#0f1115",
                   edgecolor="#444a5e", labelcolor="#e6e8ee")
    axes[-1].set_xlabel("time after stimulus (ms)")
    fig.suptitle(f"per-channel waveforms — epoch #{epoch_idx} ({State.concept_list[epoch_idx]})",
                 color="#e6e8ee", fontsize=11)
    fig.patch.set_facecolor("#161922")
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_latent_grid(epoch_idx: int, model_names: list[str]) -> str:
    panels = []
    for m in model_names:
        y_q = latent(epoch_idx, m).numpy()             # (C_lat, T_lat)
        panels.append((f"{m} latent  (max |z|={np.abs(y_q).max():.0f}, "
                       f"unique={len(np.unique(y_q))})", y_q))
    vmax = float(np.max([np.abs(p[1]).max() for p in panels]))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2*n+0.5, 4.0),
                             gridspec_kw={"wspace": 0.35})
    if n == 1: axes = [axes]
    for ax, (t, arr) in zip(axes, panels):
        im = ax.imshow(arr, aspect="auto", cmap="PuOr", vmin=-vmax, vmax=vmax)
        ax.set_title(t, fontsize=10, loc="left", color="#e6e8ee", pad=6)
        ax.set_xlabel("latent time")
        ax.set_ylabel("latent channel")
        style_axes(ax)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, label="integer symbol value")
    cb.ax.tick_params(colors="#d0d2d8")
    cb.set_label("integer symbol value", color="#bfc4d2")
    fig.patch.set_facecolor("#161922")
    return fig_to_b64(fig)


@torch.no_grad()
def fig_per_channel_mse(model_names: list[str]) -> str:
    """MSE per channel averaged across the test set."""
    x = State.eeg_avg.to(DEVICE)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    width = 0.42
    xs = np.arange(N_CHANNELS)
    for i, m in enumerate(model_names):
        c = State.codecs[m]
        xh, _ = c.compress_then_reconstruct(x)
        per_ch = (x - xh).pow(2).mean(dim=(0, 2)).cpu().numpy()
        color = "#d62728" if "neurozip" in m else "#9aa0b2"
        ax.bar(xs + (i - 0.5)*width, per_ch, width, color=color, alpha=0.85, label=m)
    ax.set_xlabel("EEG channel index"); ax.set_ylabel("mean squared error")
    ax.set_title("per-channel reconstruction error (test set, trial-averaged)",
                 fontsize=10, loc="left")
    style_axes(ax)
    ax.legend(fontsize=9, framealpha=0.4, facecolor="#0f1115",
              edgecolor="#444a5e", labelcolor="#e6e8ee")
    fig.patch.set_facecolor("#161922"); fig.tight_layout()
    return fig_to_b64(fig)


@torch.no_grad()
def fig_bit_histogram(model_names: list[str]) -> str:
    """Distribution of bits/symbol across the test set's latents."""
    x = State.eeg_avg.to(DEVICE)
    fig, ax = plt.subplots(figsize=(8, 2.4))
    for m in model_names:
        c = State.codecs[m]
        y = c.encoder(x)
        y_int = torch.round(y)
        # per-symbol bit cost
        upper = c.prior._cdf(y_int + 0.5)
        lower = c.prior._cdf(y_int - 0.5)
        p = (upper - lower).clamp_min(1e-9)
        bits = -torch.log2(p).flatten().cpu().numpy()
        color = "#d62728" if "neurozip" in m else "#9aa0b2"
        ax.hist(bits, bins=60, density=True, alpha=0.55, color=color, label=m)
    ax.set_xlabel("bits per latent symbol"); ax.set_ylabel("density")
    ax.set_title("entropy distribution across all latent positions",
                 fontsize=10, loc="left")
    style_axes(ax)
    ax.legend(fontsize=9, framealpha=0.4, facecolor="#0f1115",
              edgecolor="#444a5e", labelcolor="#e6e8ee")
    fig.patch.set_facecolor("#161922"); fig.tight_layout()
    return fig_to_b64(fig)


# ============================================================================
# API
# ============================================================================

@app.before_request
def _ensure_state():
    load_state()


def _root_html() -> str:
    """Which HTML file `/` serves. Override via env NEUROZIP_HOME=clean|dc|dark."""
    home = os.environ.get("NEUROZIP_HOME", "dark").lower()
    return {
        "clean": "demo_clean.html",
        "dc":    "NeuroZip.dc.html",
        "v2":    "NeuroZip.dc.html",
        "dark":  "demo.html",
    }.get(home, "demo.html")


@app.get("/")
def index():
    return send_file(ROOT / _root_html())


@app.get("/clean")
def index_clean():
    """White-theme alternative to demo.html — same API, different shell."""
    return send_file(ROOT / "demo_clean.html")


@app.get("/dc")
def index_dc():
    """v2 design: ported NeuroZip.dc.html with codec workspace."""
    return send_file(ROOT / "NeuroZip.dc.html")


@app.get("/dark")
def index_dark():
    """Dark-theme alternative."""
    return send_file(ROOT / "demo.html")


@app.get("/demo/<path:p>")
def demo_assets(p):
    return send_from_directory(DEMO_DIR, p)


@app.get("/api/health")
def health():
    return jsonify(ok=True, device=DEVICE, codecs=list(State.codecs))


@app.get("/api/models")
def models():
    """All codec metadata (for the tier slider)."""
    scores_path = CHECKPOINTS / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    out = {}
    for name in State.codecs:
        s = scores.get(name, {})
        out[name] = {
            "bpp": s.get("bpp"),
            "ratio": s.get("compression_ratio_vs_fp16"),
            "mse": s.get("mse"),
            "img_top1": s.get("retrieval_image", {}).get("top1"),
            "img_top5": s.get("retrieval_image", {}).get("top5"),
            "img_top10": s.get("retrieval_image", {}).get("top10"),
            "text_top1": s.get("retrieval_text", {}).get("top1"),
            "text_top5": s.get("retrieval_text", {}).get("top5"),
            "text_top10": s.get("retrieval_text", {}).get("top10"),
            "holdout_top1": s.get("holdout_top1"),
        }
    return jsonify(out)


@app.get("/api/baseline")
def baseline():
    """Reference numbers for contextualizing the codecs' retrieval scores.

    - chance: random retrieval (k/200)
    - raw_eeg: top-k from the frozen projector on uncompressed trial-averaged EEG.
              This is OUR ceiling — no codec can do better than this on the
              same retrieval metric, since every codec's output is decoded
              through the same projector.
    - sota: best published top-5 on this exact protocol (THINGS-EEG, 200-way,
            trial-averaged), as a reference for what a bigger judge / longer
            training / multi-subject would buy.
    """
    scores_path = CHECKPOINTS / "scores.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    raw = {}
    if scores:
        # Every codec records the same retrieval_raw_image (same projector + EEG).
        any_codec = next(iter(scores.values()))
        raw = any_codec.get("retrieval_raw_image", {})
    return jsonify({
        "chance":  {"top1": 1/200, "top5": 5/200, "top10": 10/200},
        "raw_eeg": {
            "top1":  raw.get("top1"),
            "top5":  raw.get("top5"),
            "top10": raw.get("top10"),
            "label": "frozen projector P on uncompressed trial-averaged EEG (our ceiling)",
        },
        "sota": {
            "top1":  0.34,
            "top5":  0.65,
            "top10": 0.79,
            "label": "approx. SOTA on THINGS-EEG 200-way trial-averaged (NICE / Wu et al. CVPR 2025; multi-subject, ~10-30M-param projector, 100+ epochs)",
        },
        "n_concepts": 200,
        "raw_fp16_bpp": 16.0,
    })


@app.get("/api/concepts")
def concepts():
    return jsonify({"concepts": State.concept_list,
                    "image_paths": State.concept_img_paths})


@app.get("/api/concept_image/<int:idx>")
def concept_image(idx):
    if idx < 0 or idx >= len(State.concept_img_paths):
        return abort(404)
    p = IMG_DIR / State.concept_img_paths[idx]
    if not p.exists(): return abort(404)
    return send_file(p)


@app.post("/api/retrieve")
def retrieve():
    body = request.get_json(force=True)
    text = body.get("text")
    concept_idx = body.get("concept_idx")
    k = int(body.get("k", 5))
    model_names = body.get("models") or list(State.codecs.keys())

    # Get query embedding.
    if text is not None and text.strip():
        text = text.strip()
        # Live CLIP text encoding.
        q = encode_text(text).to(DEVICE)
        gold_idx = None
        # Identify closest concept by text similarity (for "gold" marking on retrievals)
        sims = F.normalize(State.text_features, dim=-1).to(DEVICE) @ q
        gold_idx = int(sims.argmax().item())
        gold_score = float(sims.max().item())
        nearest_concept = State.concept_list[gold_idx]
    elif concept_idx is not None:
        gold_idx = int(concept_idx)
        q = F.normalize(State.text_features[gold_idx].float(), dim=-1).to(DEVICE)
        nearest_concept = State.concept_list[gold_idx]; gold_score = 1.0
    else:
        return jsonify(error="provide text or concept_idx"), 400

    out = {"query": {"text": text, "nearest_concept": nearest_concept,
                     "nearest_concept_idx": gold_idx,
                     "nearest_concept_score": gold_score},
           "models": {}}
    for m in model_names:
        embs = State.eeg_emb_cache[m].to(DEVICE)        # (200, 512), already L2-normalized via proj
        sims = embs @ q                                  # (200,)
        topk = sims.topk(k)
        idxs = topk.indices.cpu().tolist()
        scores = topk.values.cpu().tolist()
        out["models"][m] = {
            "top_k_eeg_idx": idxs,
            "top_k_scores": scores,
            "top_k_concepts": [State.concept_list[i] for i in idxs],
            "gold_in_top_k": gold_idx in idxs,
            "gold_rank": int((sims.argsort(descending=True) == gold_idx).nonzero(as_tuple=True)[0].item()),
        }
    return jsonify(out)


@app.post("/api/reconstruct")
def reconstruct_endpoint():
    body = request.get_json(force=True)
    epoch_idx = int(body["epoch_idx"])
    kinds = body.get("kinds", ["heatmap", "waveforms"])
    model_names = body.get("models", ["neurozip_med", "fidelity_med"])
    figs = {}
    if "heatmap" in kinds:    figs["heatmap"]    = fig_recon_heatmaps(epoch_idx, model_names)
    if "diff"    in kinds:    figs["diff"]       = fig_recon_diff(epoch_idx, model_names)
    if "waveforms" in kinds:  figs["waveforms"]  = fig_channel_waveforms(epoch_idx, model_names)
    if "latent"  in kinds:    figs["latent"]     = fig_latent_grid(epoch_idx, model_names)
    if "channel_mse" in kinds:figs["channel_mse"] = fig_per_channel_mse(model_names)
    if "bit_hist" in kinds:   figs["bit_hist"]   = fig_bit_histogram(model_names)
    # Per-epoch headline numbers
    info = {"epoch_idx": epoch_idx, "concept": State.concept_list[epoch_idx],
            "models": {}}
    raw = State.eeg_avg[epoch_idx]
    for m in model_names:
        _, xh, bits = reconstruct(epoch_idx, m)
        mse = (raw - xh).pow(2).mean().item()
        c = State.codecs[m]
        info["models"][m] = {"bpp": c.bpp_floor(bits),
                              "bits_per_symbol": bits,
                              "mse": mse,
                              "compression_ratio_vs_fp16": 16.0 / max(c.bpp_floor(bits), 1e-6)}
    return jsonify({"info": info, "figs": figs})


# ============================================================================
# Codec workspace endpoints (NeuroZip.dc.html "Compress / Decompress" panes)
# ============================================================================

def _serialize_compressed(model_name: str, latent_int: np.ndarray) -> bytes:
    """Tiny custom-format .nz file: a single .npz containing the integer
    latent, the model name, and the per-channel norm stats (so the
    decompressor doesn't need an out-of-band lookup)."""
    bio = io.BytesIO()
    np.savez_compressed(
        bio,
        model=np.array(model_name, dtype=str),
        latent=latent_int.astype(np.int16),
        norm_mean=State.norm_mean.numpy().astype(np.float32),
        norm_std=State.norm_std.numpy().astype(np.float32),
    )
    return bio.getvalue()


def _deserialize_compressed(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        return {
            "model": str(z["model"]),
            "latent": z["latent"].astype(np.int32),
            "norm_mean": z["norm_mean"].astype(np.float32),
            "norm_std": z["norm_std"].astype(np.float32),
        }


def _decode_input_eeg(body: dict) -> tuple[np.ndarray, str]:
    """Frontend gives us EITHER an epoch_idx (use the loaded test set) OR
    a base64-encoded raw payload — try .npy first, fall back to raw float
    bytes. Returns (eeg_array_63x250_float32, source_label)."""
    if body.get("epoch_idx") is not None:
        idx = int(body["epoch_idx"])
        # Inverse-normalize back to "real" EEG units so the workspace
        # mirrors what an external caller would actually feed in.
        x = State.eeg_avg[idx].numpy()
        x_real = x * State.norm_std.numpy()[:, None] + State.norm_mean.numpy()[:, None]
        return x_real.astype(np.float32), f"test epoch #{idx}"
    blob = base64.b64decode(body["bytes_b64"])
    # Try parsing as .npy (the standard EEG epoch format).
    try:
        arr = np.load(io.BytesIO(blob), allow_pickle=False)
    except Exception:
        # Fall back to a raw float32 buffer
        arr = np.frombuffer(blob, dtype=np.float32)
    arr = arr.reshape(N_CHANNELS, N_TIMES).astype(np.float32)
    return arr, "uploaded file"


@app.post("/api/compress")
def api_compress():
    """Compress an EEG epoch through one of the trained codecs.

    Request: {"epoch_idx": int}  OR  {"bytes_b64": <base64 of .npy or raw>,
                                      "model": "neurozip_v4_high"}
    Returns: codec stats + base64-encoded .nz blob for download.
    """
    body = request.get_json(force=True)
    eeg, source = _decode_input_eeg(body)
    model_name = body.get("model", "neurozip_v4_high")
    if model_name not in State.codecs:
        return jsonify(error=f"unknown model {model_name}"), 400

    # Normalize → encode → quantize → bits → roundtrip MSE.
    eeg_n = (eeg - State.norm_mean.numpy()[:, None]) / State.norm_std.numpy()[:, None]
    x = torch.from_numpy(eeg_n).unsqueeze(0).to(DEVICE)
    codec = State.codecs[model_name]
    with torch.no_grad():
        y = codec.encoder(x)
        y_int = torch.round(y)
        bits = codec.prior.bits(y_int).item()
        xh_n = codec.decoder(y_int).cpu().squeeze(0).numpy()
    mse_norm = float(((eeg_n - xh_n) ** 2).mean())

    latent = y_int.cpu().squeeze(0).int().numpy()
    blob = _serialize_compressed(model_name, latent)
    bpp = codec.bpp_floor(bits)

    return jsonify({
        "filename": f"epoch.{model_name}.nz",
        "size_bytes": len(blob),
        "raw_fp16_bytes": N_CHANNELS * N_TIMES * 2,
        "bpp": bpp,
        "bits_per_symbol": bits,
        "ratio": 16.0 / max(bpp, 1e-9),
        "mse": mse_norm,
        "latent_shape": list(latent.shape),
        "latent_unique_values": int(np.unique(latent).size),
        "source": source,
        "bytes_b64": base64.b64encode(blob).decode(),
    })


@app.post("/api/decompress")
def api_decompress():
    """Decompress a .nz blob back into a 63x250 EEG epoch.

    Request: {"bytes_b64": <base64 of .nz file>}
    Returns: shape + reconstructed EEG as .npy bytes (base64) for download.
    """
    body = request.get_json(force=True)
    blob = base64.b64decode(body["bytes_b64"])
    payload = _deserialize_compressed(blob)
    model_name = payload["model"]
    if model_name not in State.codecs:
        return jsonify(error=f"compressed blob asks for model '{model_name}' which isn't loaded"), 400

    y = torch.from_numpy(payload["latent"]).float().unsqueeze(0).to(DEVICE)
    codec = State.codecs[model_name]
    with torch.no_grad():
        xh_n = codec.decoder(y).cpu().squeeze(0).numpy()
    # Inverse-normalize back to real EEG units.
    xh = xh_n * payload["norm_std"][:, None] + payload["norm_mean"][:, None]

    bio = io.BytesIO()
    np.save(bio, xh.astype(np.float16), allow_pickle=False)
    npy_bytes = bio.getvalue()

    return jsonify({
        "filename": f"epoch.recon.npy",
        "size_bytes": len(npy_bytes),
        "shape": list(xh.shape),
        "model": model_name,
        "bytes_b64": base64.b64encode(npy_bytes).decode(),
    })


@app.get("/api/demo_epoch/<int:idx>")
def api_demo_epoch(idx):
    """Stream a real EEG epoch from the test set as a .npy file. Lets the
    workspace's 'demo epoch' dropdown actually feed a real array to /api/compress."""
    if not (0 <= idx < State.eeg_avg.size(0)):
        return abort(404)
    x = State.eeg_avg[idx].numpy()
    x_real = x * State.norm_std.numpy()[:, None] + State.norm_mean.numpy()[:, None]
    bio = io.BytesIO()
    np.save(bio, x_real.astype(np.float16), allow_pickle=False)
    bio.seek(0)
    concept = State.concept_list[idx]
    return send_file(bio, mimetype="application/octet-stream",
                     as_attachment=True, download_name=f"{concept}_idx{idx}.npy")


@app.post("/api/aggregate_figs")
def aggregate_figs():
    """Test-set aggregates (per-channel MSE, entropy hist). Cache for a tier."""
    body = request.get_json(force=True)
    kinds = body.get("kinds", ["channel_mse", "bit_hist"])
    model_names = body.get("models", ["neurozip_med", "fidelity_med"])
    figs = {}
    if "channel_mse" in kinds: figs["channel_mse"] = fig_per_channel_mse(model_names)
    if "bit_hist"   in kinds: figs["bit_hist"]    = fig_bit_histogram(model_names)
    return jsonify({"figs": figs})


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    print(f"[serve] starting on http://{args.host}:{args.port}")
    # Eager-load so first request is fast.
    with app.app_context():
        load_state()
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
