"""
benchmark_retrieval.py — the metric the MSE-only benchmark was missing.

NeuroZip's thesis is *task-aware retrieval*, not waveform fidelity: rian's
`results.md` explicitly subordinates MSE ("worst MSE deficit ~5%, best top-5
advantage +11pp — the trade is asymmetric"). So comparing codecs on MSE alone
answers the wrong question. This script compresses the trial-averaged test EEG
with each codec, then scores **top-k retrieval through the frozen projector P**
(the same judge rian trains in Stage 1) against the per-concept CLIP image
features — alongside MSE / ratio for context.

It loads either codec type (scalar `EEGCodec` or `EEGCodecRQ`) by sniffing the
saved config, exactly like serve_rqvae.py.

Run:
    python benchmark_retrieval.py \
        --models RQ-VAE=checkpoints/codec_rqvae_72x.pt \
                 scalar-matched=checkpoints/codec_fidelity_72x.pt \
                 v4=checkpoints/codec_v4_fidelity.pt
"""
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F

from data import ThingsEEG, N_CHANNELS, N_TIMES
from codec import EEGCodec
from rqvae import EEGCodecRQ
from clip_proj import load_frozen_projector, retrieval_topk

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
JUDGE = Path("checkpoints/clip_proj.pt")


def load_codec(path):
    ck = torch.load(path, weights_only=False, map_location=DEVICE)
    c = ck["config"]
    if "num_quantizers" in c:                       # RQ-VAE
        m = EEGCodecRQ(c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"],
                       codebook_size=c["codebook_size"], num_quantizers=c["num_quantizers"])
    else:                                           # scalar EEGCodec
        m = EEGCodec(c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"])
    m.to(DEVICE); m.load_state_dict(ck["model"]); m.eval()
    ratio = ck.get("final_val", {}).get("ratio")
    return m, ratio


@torch.no_grad()
def bench(model, ratio, te, judge, gallery):
    xh, _ = model.compress_then_reconstruct(te)
    mse = F.mse_loss(xh, te).item()
    var_exp = 1 - mse / te.var().item()
    embs = judge(xh)
    gold = torch.arange(embs.size(0), device=DEVICE)
    r = retrieval_topk(embs, gallery, gold, ks=(1, 5, 10))
    if ratio is None:
        ratio = model.compression_ratio() if hasattr(model, "compression_ratio") else None
    return {"mse": mse, "var_exp": var_exp, "ratio": ratio,
            "top1": r["top1"], "top5": r["top5"], "top10": r["top10"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="name=checkpoint.pt entries")
    ap.add_argument("--out", default="results/retrieval_benchmark.json")
    args = ap.parse_args()

    test_ds = ThingsEEG(split="test")
    te = test_ds.trial_averaged()[0].to(DEVICE)
    judge = load_frozen_projector(JUDGE, DEVICE)
    gallery = test_ds.concept_clip_img().to(DEVICE)

    # Reference points: the uncompressed trial-averaged EEG through the same judge.
    with torch.no_grad():
        raw = retrieval_topk(judge(te), gallery,
                             torch.arange(te.size(0), device=DEVICE), ks=(1, 5, 10))
    print(f"{'model':18s} {'ratio':>7} {'MSE':>8} {'var':>5} "
          f"{'top1':>6} {'top5':>6} {'top10':>6}")
    print(f"{'raw (uncompressed)':18s} {'1x':>7} {0.0:8.4f} {'100%':>5} "
          f"{raw['top1']*100:5.1f} {raw['top5']*100:5.1f} {raw['top10']*100:5.1f}")

    out = {"raw": raw}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model, ratio = load_codec(path)
        r = bench(model, ratio, te, judge, gallery)
        out[name] = r
        rr = f"{r['ratio']:.0f}x" if r['ratio'] else "—"
        print(f"{name:18s} {rr:>7} {r['mse']:8.4f} {r['var_exp']*100:4.0f}% "
              f"{r['top1']*100:5.1f} {r['top5']*100:5.1f} {r['top10']*100:5.1f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
