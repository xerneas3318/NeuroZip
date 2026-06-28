"""NeuroZip training entrypoint.

Subcommands (each independently runnable):
  train.py proj            -- Stage 1: train the EEG->CLIP projector P.
  train.py codec           -- Stage 2: train the rate+MSE-only baseline codec.
  train.py neurozip        -- Stage 3: warm-start codec + add task loss.
  train.py classifier      -- Stage 4: held-out concept classifier (judge-free).

Defaults are tuned for a same-day demo on a single GPU. Bump epochs for better
numbers; the qualitative result (NeuroZip > fidelity at matched bpp) shows up
quickly.
"""

from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ThingsEEG, make_loader, N_CHANNELS, N_TIMES
from clip_proj import EEGProjector, info_nce, retrieval_topk, load_frozen_projector, CLIP_DIM, WEIGHTS as PROJ_WEIGHTS
from codec import EEGCodec, quantize, compression_ratio, CHECKPOINTS

CHECKPOINTS.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Stage 1: projector ----------

def train_projector(args):
    train_ds, train_loader = make_loader("train", batch_size=args.batch, num_workers=args.workers)
    test_ds = ThingsEEG(split="test")

    model = EEGProjector(n_channels=N_CHANNELS, hidden=args.hidden,
                          out_dim=CLIP_DIM, n_attn=args.n_attn,
                          attn_heads=args.attn_heads).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    print(f"[proj] device={DEVICE}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time(); losses = []
        for eeg, clip_img, _, _ in train_loader:
            eeg, clip_img = eeg.to(DEVICE, non_blocking=True), clip_img.to(DEVICE, non_blocking=True)
            emb = model(eeg)
            loss = info_nce(emb, clip_img, temperature=args.temp)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item())
        mean_loss = np.mean(losses)

        acc = evaluate_projector(model, test_ds)
        print(f"[proj] epoch {epoch+1:2d}/{args.epochs}  loss={mean_loss:.4f}  "
              f"trial-avg top1={acc['top1']*100:5.2f}  top5={acc['top5']*100:5.2f}  "
              f"top10={acc['top10']*100:5.2f}  ({time.time()-t0:.1f}s)")
        if acc["top1"] > best:
            best = acc["top1"]
            torch.save({"model": model.state_dict(),
                        "config": {"hidden": args.hidden, "out_dim": CLIP_DIM,
                                   "n_attn": args.n_attn,
                                   "attn_heads": args.attn_heads}},
                       PROJ_WEIGHTS)
            print(f"[proj] saved -> {PROJ_WEIGHTS} (top1={best*100:.2f}%)")

    final = evaluate_projector(load_frozen_projector(PROJ_WEIGHTS, DEVICE), test_ds)
    print(f"[proj] best top1={best*100:.2f}%, final reload top1={final['top1']*100:.2f}%")


@torch.no_grad()
def evaluate_projector(model: EEGProjector, test_ds: ThingsEEG, batch: int = 200) -> dict:
    """Trial-averaged top-k retrieval against per-concept CLIP image features."""
    model.eval()
    eeg_avg, _, _ = test_ds.trial_averaged()
    gallery = test_ds.concept_clip_img().to(DEVICE)
    embs = []
    for i in range(0, eeg_avg.size(0), batch):
        x = eeg_avg[i:i+batch].to(DEVICE)
        embs.append(model(x))
    embs = torch.cat(embs, dim=0)
    gold = torch.arange(embs.size(0), device=DEVICE)
    return retrieval_topk(embs, gallery, gold, ks=(1, 5, 10))


# ---------- Stage 2 / Stage 3: codec ----------

def train_codec(args):
    """Stage 2 (lambda_task=0) or Stage 3 (lambda_task>0).

    Stage 3 warm-starts from a Stage-2 checkpoint and adds the task term that
    routes gradient through the FROZEN projector P. Asserting non-zero decoder
    gradient from the task-only backward is the #1 way to catch a broken
    setup; we do that assert below.
    """
    out_name = args.out                            # checkpoints/<name>.pt
    out_path = CHECKPOINTS / f"{out_name}.pt"
    log_path = CHECKPOINTS / f"{out_name}.json"

    train_ds, train_loader = make_loader("train", batch_size=args.batch, num_workers=args.workers)
    test_ds = ThingsEEG(split="test")

    model = EEGCodec(c_lat=args.c_lat, hidden=args.hidden,
                      n_attn=args.n_attn).to(DEVICE)
    if args.init_from:
        state = torch.load(CHECKPOINTS / f"{args.init_from}.pt", weights_only=False, map_location=DEVICE)
        init_cfg = state.get("config", {})
        # Rebuild model with the init checkpoint's architecture, then optionally
        # widen later. For now require matching n_attn between init and out.
        if init_cfg.get("n_attn", 0) != args.n_attn:
            raise SystemExit(f"--init_from {args.init_from} has n_attn={init_cfg.get('n_attn',0)} "
                             f"but --n_attn={args.n_attn}; train a matching fidelity baseline first")
        model.load_state_dict(state["model"])
        print(f"[codec] warm-started from {args.init_from} (n_attn={args.n_attn})")

    judge = None
    if args.lambda_task > 0:
        judge = load_frozen_projector(PROJ_WEIGHTS, DEVICE)
        # Gradient flow sanity check BEFORE training (one of the 3 easy-to-
        # get-wrong spots).
        _grad_flow_assert(model, judge)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    print(f"[codec] device={DEVICE}  c_lat={args.c_lat}  lambda_task={args.lambda_task}  "
          f"lambda_rate={args.lambda_rate}")
    history = []
    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        sums = {"mse": [], "bits": [], "task": [], "loss": []}
        for eeg, clip_img, _, _ in train_loader:
            eeg, clip_img = eeg.to(DEVICE, non_blocking=True), clip_img.to(DEVICE, non_blocking=True)
            eeg_hat, bits_per_symbol, _ = model(eeg)
            mse = F.mse_loss(eeg_hat, eeg)
            loss = args.lambda_recon * mse + args.lambda_rate * bits_per_symbol
            task_val = torch.tensor(0.0, device=DEVICE)
            if judge is not None:
                # CRITICAL: do NOT wrap judge(eeg_hat) in no_grad. The CLIP
                # target IS detached (it's a constant), but the judge of the
                # reconstruction must carry gradient back into the decoder.
                eeg_emb = judge(eeg_hat)                  # gradient flows
                with torch.no_grad():
                    clip_tgt = F.normalize(clip_img, dim=-1)
                if args.task_loss == "cosine":
                    task_val = 1.0 - F.cosine_similarity(eeg_emb, clip_tgt, dim=-1).mean()
                else:  # 'infonce'
                    logits = (eeg_emb @ clip_tgt.t()) / 0.07
                    tgts = torch.arange(eeg_emb.size(0), device=DEVICE)
                    task_val = 0.5 * (F.cross_entropy(logits, tgts) + F.cross_entropy(logits.t(), tgts))
                loss = loss + args.lambda_task * task_val
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            sums["mse"].append(mse.item()); sums["bits"].append(bits_per_symbol.item())
            sums["task"].append(task_val.item()); sums["loss"].append(loss.item())
        bps = float(np.mean(sums["bits"]))
        bpp = model.bpp_floor(bps)
        ratio = compression_ratio(model, bps)
        eval_metrics = evaluate_codec(model, test_ds, judge_path=PROJ_WEIGHTS)
        rec = {"epoch": epoch + 1,
               "loss": float(np.mean(sums["loss"])),
               "mse": float(np.mean(sums["mse"])),
               "bits_per_symbol": bps,
               "bpp": float(bpp),
               "compression_ratio_vs_fp16": float(ratio),
               "task": float(np.mean(sums["task"])),
               **eval_metrics,
               "epoch_seconds": time.time() - t0}
        history.append(rec)
        print(f"[codec:{out_name}] e{epoch+1:2d}/{args.epochs}  "
              f"loss={rec['loss']:.3f}  mse={rec['mse']:.4f}  "
              f"bits/sym={bps:.3f}  bpp={bpp:.3f}  ratio={ratio:.1f}x  "
              f"top1={eval_metrics['top1']*100:5.2f}  top5={eval_metrics['top5']*100:5.2f}  "
              f"({rec['epoch_seconds']:.1f}s)")

    torch.save({"model": model.state_dict(),
                "config": {"c_lat": args.c_lat, "hidden": args.hidden,
                           "n_attn": args.n_attn,
                           "lambda_task": args.lambda_task,
                           "lambda_recon": args.lambda_recon,
                           "lambda_rate": args.lambda_rate,
                           "task_loss": args.task_loss}}, out_path)
    log_path.write_text(json.dumps(history, indent=2))
    print(f"[codec:{out_name}] saved {out_path} ; log {log_path}")


def _grad_flow_assert(model: EEGCodec, judge: EEGProjector):
    """Stage-3 sanity: confirm gradient propagates from task term -> decoder."""
    model.train()
    x = torch.randn(4, N_CHANNELS, N_TIMES, device=DEVICE, requires_grad=False)
    # Ensure decoder grads start fresh.
    for p in model.decoder.parameters():
        if p.grad is not None: p.grad = None
    eeg_hat, _, _ = model(x)
    emb = judge(eeg_hat)
    fake_target = F.normalize(torch.randn_like(emb), dim=-1)
    (1.0 - F.cosine_similarity(emb, fake_target, dim=-1).mean()).backward()
    nonzero = sum((p.grad is not None and p.grad.abs().sum().item() > 0)
                  for p in model.decoder.parameters())
    assert nonzero > 0, "task loss did not propagate into decoder weights!"
    # Judge must NOT have gotten any grad.
    judge_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                     for p in judge.parameters())
    assert not judge_grad, "frozen judge received gradient updates!"
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    print(f"[codec] grad-flow OK: decoder params with non-zero grad = {nonzero}")


@torch.no_grad()
def evaluate_codec(model: EEGCodec, test_ds: ThingsEEG, judge_path: Path,
                   batch: int = 200) -> dict:
    """Compress test EEG, then judge with frozen P.

    Returns trial-averaged top-k retrieval against per-concept CLIP image
    features, plus reconstruction MSE on normalized EEG, plus inference bpp.
    """
    model.eval()
    judge = load_frozen_projector(judge_path, DEVICE)
    eeg_avg, _, _ = test_ds.trial_averaged()
    gallery = test_ds.concept_clip_img().to(DEVICE)
    embs, mses, bits_list = [], [], []
    for i in range(0, eeg_avg.size(0), batch):
        x = eeg_avg[i:i+batch].to(DEVICE)
        eeg_hat, bits = model.compress_then_reconstruct(x)
        mses.append(F.mse_loss(eeg_hat, x).item())
        bits_list.append(bits.item())
        embs.append(judge(eeg_hat))
    embs = torch.cat(embs, dim=0)
    gold = torch.arange(embs.size(0), device=DEVICE)
    metrics = retrieval_topk(embs, gallery, gold, ks=(1, 5, 10))
    metrics["mse_inference"] = float(np.mean(mses))
    metrics["bits_per_symbol_inference"] = float(np.mean(bits_list))
    metrics["bpp_inference"] = float(model.bpp_floor(metrics["bits_per_symbol_inference"]))
    return metrics


# ---------- Stage 4: held-out concept classifier (circularity defense) ----------

class HoldoutClassifier(nn.Module):
    """A separate EEG->concept classifier trained ON RAW EEG (never sees the codec).

    Used in evaluate.py to score retained accuracy of NeuroZip's reconstructions
    with a judge the compressor's loss never optimized against. If raw retrieval
    accuracy of NeuroZip-decompressed EEG is still high under THIS judge, the
    Stage-3 numbers aren't just gaming the projector.
    """

    def __init__(self, n_classes: int, hidden: int = 128, n_attn: int = 0,
                 attn_heads: int = 4):
        super().__init__()
        self.body = EEGProjector(hidden=hidden, out_dim=hidden * 2,
                                  n_attn=n_attn, attn_heads=attn_heads)
        self.cls = nn.Linear(hidden * 2, n_classes)

    def forward(self, eeg):
        h = self.body(eeg)                 # already L2-normalized; that's fine for classify
        return self.cls(h)


def train_holdout_classifier(args):
    """Train an EEG->concept classifier on the TEST split's repetitions.

    We split the 80 reps per concept into 60 train / 20 eval. To get a usable
    SNR we re-average groups of K=5 reps each iteration: the codec sees an
    average over 80 reps at inference time, so training on cleaner averaged
    chunks (not raw single trials) is much more sample-efficient.

    Crucially: this classifier never sees the codec's output. It is judged on
    decompressed-EEG only AFTER training, in evaluate.py, which is the
    circularity defense.
    """
    test_ds = ThingsEEG(split="test")
    eeg = test_ds.eeg.reshape(test_ds.n_concepts, test_ds.n_reps, N_CHANNELS, N_TIMES)
    rng = np.random.default_rng(0)
    perm = rng.permutation(test_ds.n_reps)
    tr_idx, va_idx = perm[:60], perm[60:]
    K = args.group        # reps to average per training sample

    n_cls = test_ds.n_concepts
    # Pre-average eval set across all 20 held-out reps -> one sample per concept,
    # which is what the codec's trial-averaged output also looks like (avg over reps).
    xva = eeg[:, va_idx].mean(axis=1)
    yva = np.arange(n_cls)
    xva_t = torch.from_numpy(xva).float().to(DEVICE)
    yva_t = torch.from_numpy(yva).long().to(DEVICE)

    def sample_train_batch(bs: int):
        # For each of `bs` samples: pick a random concept, pick K reps, average.
        idxs = rng.integers(0, n_cls, size=bs)
        out = np.empty((bs, N_CHANNELS, N_TIMES), dtype=np.float32)
        for i, ci in enumerate(idxs):
            r = rng.choice(tr_idx, size=K, replace=False)
            out[i] = eeg[ci, r].mean(axis=0)
        return torch.from_numpy(out), torch.from_numpy(idxs).long()
    xtr_t = None; ytr_t = None  # not used; we sample on the fly

    model = HoldoutClassifier(n_classes=n_cls, hidden=args.hidden,
                                n_attn=args.n_attn, attn_heads=args.attn_heads).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    steps_per_epoch = args.steps_per_epoch
    bs = args.batch
    best = -1.0
    for epoch in range(args.epochs):
        model.train(); losses = []
        for _ in range(steps_per_epoch):
            xb, yb = sample_train_batch(bs)
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(xva_t), 256):
                preds.append(model(xva_t[i:i+256]).argmax(dim=-1))
            preds = torch.cat(preds)
            acc = (preds == yva_t).float().mean().item()
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "n_classes": n_cls,
                        "hidden": args.hidden,
                        "n_attn": args.n_attn, "attn_heads": args.attn_heads},
                       CHECKPOINTS / "holdout_classifier.pt")
        print(f"[holdout] epoch {epoch+1}/{args.epochs}  loss={np.mean(losses):.3f}  "
              f"eval-top1={acc*100:.2f}  best={best*100:.2f}  (K={K})")
    print(f"[holdout] best top1 = {best*100:.2f}%  saved to "
          f"{CHECKPOINTS/'holdout_classifier.pt'}")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("proj", help="Stage 1: train EEG->CLIP projector P.")
    a.add_argument("--epochs", type=int, default=6)
    a.add_argument("--batch", type=int, default=512)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--hidden", type=int, default=128)
    a.add_argument("--temp", type=float, default=0.07)
    a.add_argument("--n_attn", type=int, default=0,
                   help="transformer blocks on the [CLS] head (0 = original avgpool+MLP)")
    a.add_argument("--attn_heads", type=int, default=4)
    a.add_argument("--workers", type=int, default=2)
    a.set_defaults(func=train_projector)

    a = sub.add_parser("codec", help="Stage 2: train fidelity-only codec.")
    a.add_argument("--epochs", type=int, default=4)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--hidden", type=int, default=128)
    a.add_argument("--c_lat", type=int, default=32)
    a.add_argument("--n_attn", type=int, default=0,
                   help="transformer blocks at codec bottleneck (0 = conv-only)")
    a.add_argument("--lambda_recon", type=float, default=1.0)
    a.add_argument("--lambda_rate", type=float, default=0.01)
    a.add_argument("--lambda_task", type=float, default=0.0)
    a.add_argument("--task_loss", choices=["cosine", "infonce"], default="cosine")
    a.add_argument("--init_from", type=str, default=None,
                   help="checkpoint name to warm-start from (no .pt suffix)")
    a.add_argument("--out", type=str, default="codec_fidelity")
    a.add_argument("--workers", type=int, default=2)
    a.set_defaults(func=train_codec)

    a = sub.add_parser("neurozip", help="Stage 3: warm-start codec + task loss.")
    a.add_argument("--epochs", type=int, default=4)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--lr", type=float, default=2e-4)
    a.add_argument("--hidden", type=int, default=128)
    a.add_argument("--c_lat", type=int, default=32)
    a.add_argument("--n_attn", type=int, default=0,
                   help="transformer blocks at codec bottleneck (must match init_from)")
    a.add_argument("--lambda_recon", type=float, default=1.0)
    a.add_argument("--lambda_rate", type=float, default=0.01)
    a.add_argument("--lambda_task", type=float, default=1.0)
    a.add_argument("--task_loss", choices=["cosine", "infonce"], default="cosine")
    a.add_argument("--init_from", type=str, default="codec_fidelity")
    a.add_argument("--out", type=str, default="codec_neurozip")
    a.add_argument("--workers", type=int, default=2)
    a.set_defaults(func=train_codec)

    a = sub.add_parser("classifier", help="Stage 4: held-out concept classifier.")
    a.add_argument("--epochs", type=int, default=20)
    a.add_argument("--steps_per_epoch", type=int, default=200)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--lr", type=float, default=3e-4)
    a.add_argument("--hidden", type=int, default=128)
    a.add_argument("--n_attn", type=int, default=0,
                   help="transformer blocks on the body's [CLS] head")
    a.add_argument("--attn_heads", type=int, default=4)
    a.add_argument("--group", type=int, default=10,
                   help="K: how many train reps to average per training sample")
    a.set_defaults(func=train_holdout_classifier)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
