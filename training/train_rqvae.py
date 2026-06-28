"""Train the RQ-VAE EEG codec (residual vector-quantized, task-aware).

Mirrors train.py's codec trainer but for EEGRQVAE: loss = recon + lambda_vq * vq_loss
(+ lambda_task * task through the frozen projector). Bitrate is fixed by (n_q, K).
Optionally warm-starts the conv encoder/decoder from a trained continuous codec.

Example (on the desk box, with the dataset):
  python train_rqvae.py --epochs 6 --n_q 8 --codebook 512 \
      --lambda_task 1.0 --init_from fidelity_v4_high --out rqvae_high
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from data import make_loader, ThingsEEG, N_CHANNELS, N_TIMES
from clip_proj import load_frozen_projector, retrieval_topk
from codec import CHECKPOINTS
from rqvae import EEGRQVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJ_WEIGHTS = CHECKPOINTS / "clip_proj.pt"


@torch.no_grad()
def evaluate(model: EEGRQVAE, test_ds: ThingsEEG, judge) -> dict:
    eeg_avg, _texts, _imgs = test_ds.trial_averaged()        # (200,63,250)
    img_feats = test_ds.concept_clip_img()                   # (200,512)
    x = eeg_avg.to(DEVICE)
    xh, _ = model.compress_then_reconstruct(x)
    emb = judge(xh).cpu()
    mse = F.mse_loss(xh, x).item()
    gold = torch.arange(emb.size(0))
    r = retrieval_topk(emb, img_feats, gold, ks=(1, 5, 10))
    return {"mse": mse, "top1": r["top1"], "top5": r["top5"], "top10": r["top10"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--c_lat", type=int, default=32)
    ap.add_argument("--n_attn", type=int, default=0)
    ap.add_argument("--n_q", type=int, default=8, help="residual VQ stages")
    ap.add_argument("--codebook", type=int, default=512, help="codes per stage (K)")
    ap.add_argument("--beta", type=float, default=0.25, help="commitment weight")
    ap.add_argument("--lambda_recon", type=float, default=1.0)
    ap.add_argument("--lambda_vq", type=float, default=1.0)
    ap.add_argument("--lambda_task", type=float, default=1.0)
    ap.add_argument("--task_loss", choices=["cosine", "infonce"], default="cosine")
    ap.add_argument("--init_from", type=str, default=None,
                    help="continuous codec checkpoint to warm-start encoder/decoder")
    ap.add_argument("--out", type=str, default="rqvae_high")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    out_path = CHECKPOINTS / f"{args.out}.pt"
    log_path = CHECKPOINTS / f"{args.out}.json"
    train_ds, train_loader = make_loader("train", batch_size=args.batch, num_workers=args.workers)
    test_ds = ThingsEEG(split="test")

    model = EEGRQVAE(c_lat=args.c_lat, hidden=args.hidden, n_attn=args.n_attn,
                     n_q=args.n_q, codebook_size=args.codebook, beta=args.beta).to(DEVICE)

    if args.init_from:
        st = torch.load(CHECKPOINTS / f"{args.init_from}.pt", weights_only=False, map_location=DEVICE)
        enc_dec = {k: v for k, v in st["model"].items()
                   if k.startswith("encoder.") or k.startswith("decoder.")}
        missing, unexpected = model.load_state_dict(enc_dec, strict=False)
        print(f"[rqvae] warm-started encoder/decoder from {args.init_from} "
              f"({len(enc_dec)} tensors; {len(missing)} left random)")

    judge = load_frozen_projector(PROJ_WEIGHTS, DEVICE) if args.lambda_task > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    bpp = model.bpp_floor()
    print(f"[rqvae] device={DEVICE} n_q={args.n_q} K={args.codebook} "
          f"-> fixed bpp={bpp:.4f} ratio={16.0/bpp:.1f}x  lambda_task={args.lambda_task}")
    history = []
    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        s = {"mse": [], "vq": [], "task": [], "loss": []}
        for eeg, clip_img, _, _ in train_loader:
            eeg, clip_img = eeg.to(DEVICE, non_blocking=True), clip_img.to(DEVICE, non_blocking=True)
            eeg_hat, vq_loss, _bits, _codes = model(eeg)
            mse = F.mse_loss(eeg_hat, eeg)
            loss = args.lambda_recon * mse + args.lambda_vq * vq_loss
            task_val = torch.tensor(0.0, device=DEVICE)
            if judge is not None:
                emb = judge(eeg_hat)                       # gradient flows into decoder
                with torch.no_grad():
                    tgt = F.normalize(clip_img, dim=-1)
                if args.task_loss == "cosine":
                    task_val = 1.0 - F.cosine_similarity(emb, tgt, dim=-1).mean()
                else:
                    logits = (emb @ tgt.t()) / 0.07
                    lab = torch.arange(emb.size(0), device=DEVICE)
                    task_val = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
                loss = loss + args.lambda_task * task_val
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            s["mse"].append(mse.item()); s["vq"].append(float(vq_loss))
            s["task"].append(float(task_val)); s["loss"].append(loss.item())
        ev = evaluate(model, test_ds, load_frozen_projector(PROJ_WEIGHTS, DEVICE))
        rec = {"epoch": epoch + 1, "loss": float(np.mean(s["loss"])),
               "mse": float(np.mean(s["mse"])), "vq": float(np.mean(s["vq"])),
               "task": float(np.mean(s["task"])), "bpp": float(bpp),
               "compression_ratio_vs_fp16": float(16.0 / bpp),
               **ev, "epoch_seconds": time.time() - t0}
        history.append(rec)
        print(f"[rqvae:{args.out}] e{epoch+1:2d}/{args.epochs} loss={rec['loss']:.3f} "
              f"mse={rec['mse']:.4f} vq={rec['vq']:.4f} bpp={bpp:.4f} "
              f"top1={ev['top1']*100:5.2f} top5={ev['top5']*100:5.2f} ({rec['epoch_seconds']:.1f}s)")

    torch.save({"model": model.state_dict(),
                "config": {"model_type": "rqvae", "c_lat": args.c_lat, "hidden": args.hidden,
                           "n_attn": args.n_attn, "n_q": args.n_q, "codebook_size": args.codebook,
                           "beta": args.beta, "lambda_task": args.lambda_task,
                           "lambda_vq": args.lambda_vq, "task_loss": args.task_loss}}, out_path)
    log_path.write_text(json.dumps(history, indent=2))
    print(f"[rqvae:{args.out}] saved {out_path} ; log {log_path}")


if __name__ == "__main__":
    main()
