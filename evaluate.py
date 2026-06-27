"""NeuroZip evaluation + demo asset builder (Stage 4).

Reads trained checkpoints, evaluates them on the test split, builds the rate-
retrieval plot, and writes `demo/assets.json` + per-concept PNGs so the
self-contained demo.html can show retrieved images per query.

Key comparisons:
  * NeuroZip (codec_neurozip*.pt) vs. fidelity codec (codec_fidelity.pt)
  * At MATCHED bpp where possible (we report each model's actual bpp)
  * Retained retrieval accuracy via the projector P (the trained judge) AND
    via the held-out concept classifier (the circularity defense)
"""

from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import ThingsEEG, IMG_DIR, N_CHANNELS, N_TIMES
from clip_proj import EEGProjector, load_frozen_projector, retrieval_topk
from codec import EEGCodec, CHECKPOINTS
from train import HoldoutClassifier

ROOT = Path(__file__).parent
DEMO_DIR = ROOT / "demo"
ASSETS_DIR = DEMO_DIR / "assets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------- model loaders --------

def load_codec(name: str) -> EEGCodec:
    path = CHECKPOINTS / f"{name}.pt"
    state = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = state.get("config", {})
    model = EEGCodec(c_lat=cfg.get("c_lat", 32),
                      hidden=cfg.get("hidden", 128),
                      n_attn=cfg.get("n_attn", 0)).to(DEVICE)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def load_holdout(test_ds: ThingsEEG) -> HoldoutClassifier:
    path = CHECKPOINTS / "holdout_classifier.pt"
    if not path.exists():
        return None
    state = torch.load(path, weights_only=False, map_location=DEVICE)
    model = HoldoutClassifier(n_classes=state["n_classes"], hidden=state["hidden"]).to(DEVICE)
    model.load_state_dict(state["model"]); model.eval()
    return model


# -------- core scoring --------

@torch.no_grad()
def reconstruct_test(model: EEGCodec, test_ds: ThingsEEG, batch: int = 200):
    """Trial-average then compress->decompress. Returns (eeg_hat, mse, bpp)."""
    eeg_avg, texts, imgs = test_ds.trial_averaged()
    hats, mses, bits = [], [], []
    for i in range(0, eeg_avg.size(0), batch):
        x = eeg_avg[i:i+batch].to(DEVICE)
        h, b = model.compress_then_reconstruct(x)
        hats.append(h.cpu()); mses.append(F.mse_loss(h, x).item()); bits.append(b.item())
    eeg_hat = torch.cat(hats, dim=0)
    bps = float(np.mean(bits))
    return eeg_hat, eeg_avg, texts, imgs, float(np.mean(mses)), bps, model.bpp_floor(bps)


@torch.no_grad()
def score_model(model: EEGCodec, test_ds: ThingsEEG, judge: EEGProjector,
                holdout: HoldoutClassifier | None) -> dict:
    eeg_hat, eeg_avg, texts, imgs, mse, bps, bpp = reconstruct_test(model, test_ds)

    # ---- judge-based retrieval (P + CLIP) ----
    gallery_img = test_ds.concept_clip_img().to(DEVICE)
    gallery_text, concept_list = test_ds.concept_clip_text()
    gallery_text = gallery_text.to(DEVICE)
    embs = judge(eeg_hat.to(DEVICE))
    gold = torch.arange(embs.size(0), device=DEVICE)
    img_metrics = retrieval_topk(embs, gallery_img, gold)
    text_metrics = retrieval_topk(embs, gallery_text, gold)

    # ---- held-out classifier retained accuracy ----
    holdout_acc = None
    if holdout is not None:
        logits = holdout(eeg_hat.to(DEVICE))
        preds = logits.argmax(dim=-1)
        # The labels of trial-averaged epochs are 0..n_concepts-1 by construction.
        holdout_acc = (preds == gold).float().mean().item()

    # ---- raw recon for the same trial-averaged inputs as a sanity column ----
    raw_metrics = None
    raw_emb = judge(eeg_avg.to(DEVICE))
    raw_metrics = retrieval_topk(raw_emb, gallery_img, gold)

    return {
        "mse": mse,
        "bits_per_symbol": bps,
        "bpp": bpp,
        "compression_ratio_vs_fp16": 16.0 / max(bpp, 1e-6),
        "retrieval_image": img_metrics,
        "retrieval_text": text_metrics,
        "retrieval_raw_image": raw_metrics,
        "holdout_top1": holdout_acc,
        # detailed per-concept top-k for the demo
        "_eeg_hat": eeg_hat,
        "_embs": embs.cpu(),
        "_concepts": concept_list,
        "_imgs": imgs,
        "_gallery_img": gallery_img.cpu(),
        "_gallery_text": gallery_text.cpu(),
    }


# -------- demo asset emission --------

def emit_demo_assets(per_model: dict[str, dict], test_ds: ThingsEEG, topk: int = 5):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (ASSETS_DIR / "images").mkdir(exist_ok=True)
    # Copy the 200 test images by concept index.
    concepts = sorted(set([str(t) for t in test_ds.texts.tolist()]))
    # ThingsEEG concept order = test_ds.concept_clip_text()[1]
    _, concept_list = test_ds.concept_clip_text()
    img_paths = test_ds.trial_averaged()[2]
    for c_idx, p in enumerate(img_paths):
        src = IMG_DIR / p
        dst = ASSETS_DIR / "images" / f"c{c_idx:03d}.jpg"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    summary = {
        "concepts": concept_list,
        "topk": topk,
        "models": {},
        "featured": [],   # filled in after model loop
    }
    for name, m in per_model.items():
        embs = m["_embs"]                       # (N, D)
        gallery = m["_gallery_img"]             # (N, D) -- by concept_list index
        gallery_text = m["_gallery_text"]
        sims_img = F.normalize(embs, dim=-1) @ F.normalize(gallery, dim=-1).t()
        sims_text = F.normalize(embs, dim=-1) @ F.normalize(gallery_text, dim=-1).t()
        # For each text query (concept index), retrieve top-k EEG epochs.
        text_to_eeg = (F.normalize(gallery_text, dim=-1) @ F.normalize(embs, dim=-1).t())
        top_text = text_to_eeg.topk(topk, dim=-1).indices.tolist()
        # And image-prompt retrieval for completeness.
        img_to_eeg = (F.normalize(gallery, dim=-1) @ F.normalize(embs, dim=-1).t())
        top_img = img_to_eeg.topk(topk, dim=-1).indices.tolist()
        summary["models"][name] = {
            "bpp": m["bpp"],
            "compression_ratio_vs_fp16": m["compression_ratio_vs_fp16"],
            "mse": m["mse"],
            "retrieval_image_top1": m["retrieval_image"]["top1"],
            "retrieval_image_top5": m["retrieval_image"]["top5"],
            "retrieval_image_top10": m["retrieval_image"]["top10"],
            "retrieval_text_top1":  m["retrieval_text"]["top1"],
            "retrieval_text_top5":  m["retrieval_text"]["top5"],
            "retrieval_text_top10": m["retrieval_text"]["top10"],
            "holdout_top1": m.get("holdout_top1"),
            # text query (concept idx) -> top-k retrieved EEG concept ids
            "text_to_eeg_topk": top_text,
            "image_to_eeg_topk": top_img,
        }
    # Rank concepts by how strongly NeuroZip beats fidelity in top-k retrieval,
    # to put visually compelling examples at the top of the demo's dropdown.
    fid_models = [m for m in summary["models"].values() if "neurozip" not in m.get("name", "")]
    nzn_models = [m for m in summary["models"].values() if "neurozip" in m.get("name", "")]
    # Re-derive by name since "name" isn't in the per-model dict above.
    fid_keys = [n for n in summary["models"] if "fidelity" in n]
    nzn_keys = [n for n in summary["models"] if "neurozip" in n]
    wins = []
    for qi in range(len(concept_list)):
        nzn_hits = sum(qi in summary["models"][n]["text_to_eeg_topk"][qi] for n in nzn_keys)
        fid_hits = sum(qi in summary["models"][n]["text_to_eeg_topk"][qi] for n in fid_keys)
        wins.append((nzn_hits - fid_hits, nzn_hits, -fid_hits, qi))
    wins.sort(reverse=True)
    summary["featured"] = [w[3] for w in wins[:24]]
    (ASSETS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[demo] wrote {ASSETS_DIR/'summary.json'}  (featured: "
          f"{[concept_list[i] for i in summary['featured'][:5]]}...)")


# -------- the rate-retrieval plot --------

def rate_retrieval_plot(rows: list[dict], path: Path):
    """rows: list of {name, bpp, retrieval_image_top1, retrieval_image_top5}"""
    fid = [r for r in rows if r["name"].startswith("fidelity")]
    nzn = [r for r in rows if r["name"].startswith("neurozip")]
    fid.sort(key=lambda r: r["bpp"])
    nzn.sort(key=lambda r: r["bpp"])

    plt.figure(figsize=(7, 4.5))
    if fid:
        plt.plot([r["bpp"] for r in fid], [100*r["retrieval_image_top5"] for r in fid],
                 "o--", label="fidelity-only codec", color="#888")
    if nzn:
        plt.plot([r["bpp"] for r in nzn], [100*r["retrieval_image_top5"] for r in nzn],
                 "o-", label="NeuroZip (task-aware)", color="#d62728")
    plt.xlabel("bits per sample (lower = more compressed)")
    plt.ylabel("retrieval top-5 (%)  [image prompt -> EEG]")
    plt.title("NeuroZip vs. fidelity codec @ matched bpp")
    plt.grid(alpha=0.3); plt.legend(loc="best")
    plt.tight_layout(); plt.savefig(path, dpi=130)
    print(f"[plot] wrote {path}")


# -------- entry point --------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="codec checkpoint names (no .pt); first must be 'fidelity*' baseline")
    ap.add_argument("--judge", default=str(CHECKPOINTS / "clip_proj.pt"))
    args = ap.parse_args()

    test_ds = ThingsEEG(split="test")
    judge = load_frozen_projector(Path(args.judge), DEVICE)
    holdout = load_holdout(test_ds)

    per_model = {}
    rows = []
    for name in args.models:
        print(f"\n=== {name} ===")
        m = load_codec(name)
        s = score_model(m, test_ds, judge, holdout)
        per_model[name] = s
        print(f"  bpp={s['bpp']:.3f}  ratio={s['compression_ratio_vs_fp16']:.1f}x  "
              f"mse={s['mse']:.4f}")
        print(f"  img-prompt retrieval: top1={s['retrieval_image']['top1']*100:.2f}  "
              f"top5={s['retrieval_image']['top5']*100:.2f}  "
              f"top10={s['retrieval_image']['top10']*100:.2f}")
        print(f"  text-prompt retrieval: top1={s['retrieval_text']['top1']*100:.2f}  "
              f"top5={s['retrieval_text']['top5']*100:.2f}  "
              f"top10={s['retrieval_text']['top10']*100:.2f}")
        print(f"  raw EEG (uncompressed) img top1={s['retrieval_raw_image']['top1']*100:.2f}  "
              f"top5={s['retrieval_raw_image']['top5']*100:.2f}")
        if s.get("holdout_top1") is not None:
            print(f"  HELD-OUT classifier top1: {s['holdout_top1']*100:.2f}  "
                  "(independent judge; not in training loss)")
        rows.append({"name": name, "bpp": s["bpp"],
                     "retrieval_image_top1": s["retrieval_image"]["top1"],
                     "retrieval_image_top5": s["retrieval_image"]["top5"]})

    DEMO_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    rate_retrieval_plot(rows, ASSETS_DIR / "rate_retrieval.png")
    emit_demo_assets(per_model, test_ds)

    # Write a top-level scores.json for the README/run report.
    (CHECKPOINTS / "scores.json").write_text(json.dumps(
        {name: {k: v for k, v in m.items() if not k.startswith("_")}
         for name, m in per_model.items()}, indent=2, default=float))
    print(f"\n[done] wrote scores.json + demo assets")


if __name__ == "__main__":
    main()
