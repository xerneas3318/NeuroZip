"""
fable_train.py -- train the 3D VQ-VAE codec on Haxby 4D fMRI volumes.

Loss = masked reconstruction MSE (in-brain only) + VQ commitment.
The background is a known-zero constant (side-info), so we supervise and score
only in-brain voxels -- that's the actual functional signal.

    python fable_train.py --n-down 3 --K 512 --epochs 120 --out checkpoints/fable_nd3.pt
"""
from __future__ import annotations
import argparse, time, math, os
import numpy as np
import torch
import torch.nn.functional as F

from fable_codec import VQCodec, count_params
import fable_data


def masked_mse(xh, x, mask):
    d = (xh - x) ** 2
    return (d * mask).sum() / (mask.sum() * x.shape[0])


@torch.no_grad()
def evaluate(codec, X, mask, bs=128):
    codec.eval()
    dev = X.device
    se = 0.0
    n_vox = mask.sum().item()
    # accumulate for voxelwise temporal correlation
    sx = sy = sxx = syy = sxy = None
    codes_used = set()
    for i in range(0, X.shape[0], bs):
        x = X[i:i + bs]
        codes = codec.encode(x)
        xh = codec.decode(codes)
        se += ((xh - x) ** 2 * mask).sum().item()
        codes_used.update(codes.reshape(-1).tolist())
        xb = x[:, 0][:, mask]           # (b, nvox)
        yb = xh[:, 0][:, mask]
        s = [xb.sum(0), yb.sum(0), (xb * xb).sum(0), (yb * yb).sum(0), (xb * yb).sum(0)]
        if sx is None:
            sx, sy, sxx, syy, sxy = s
        else:
            sx += s[0]; sy += s[1]; sxx += s[2]; syy += s[3]; sxy += s[4]
    N = X.shape[0]
    mse = se / (n_vox * N)
    var = (X[:, 0][:, mask].var()).item()
    ve = 1 - mse / var
    # per-voxel temporal correlation, averaged over voxels
    cov = sxy / N - (sx / N) * (sy / N)
    vx = sxx / N - (sx / N) ** 2
    vy = syy / N - (sy / N) ** 2
    corr = (cov / (vx.sqrt() * vy.sqrt() + 1e-8)).mean().item()
    return dict(mse=mse, ve=ve, corr=corr, used=len(codes_used))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n-down", type=int, default=3)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--commit-beta", type=float, default=0.25)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out", default="checkpoints/fable_codec.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    dev = torch.device(args.device)

    d = fable_data.load()
    mask = torch.from_numpy(d["mask"]).to(dev)                    # (D,H,W) bool
    tr = torch.from_numpy(d["train"]).unsqueeze(1).to(dev)        # (N,1,D,H,W)
    te = torch.from_numpy(d["test"]).unsqueeze(1).to(dev)
    print(f"device={dev}  train={len(tr)} test={len(te)}  vol={fable_data.VOL_SHAPE}")

    codec = VQCodec(vol_shape=fable_data.VOL_SHAPE, hidden=args.hidden, dim=args.dim,
                    n_down=args.n_down, depth=args.depth, K=args.K,
                    commit_beta=args.commit_beta).to(dev)
    raw_bpv = codec.raw_bits_per_vol()
    print(f"VQCodec params={count_params(codec)/1e6:.2f}M  grid={codec.grid} "
          f"tokens/vol={codec.tokens_per_vol} K={args.K}  raw={raw_bpv/8:.0f} B/vol")

    opt = torch.optim.AdamW(codec.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    gen = torch.Generator(device=dev).manual_seed(0)
    N = len(tr)
    best = -1e9
    for ep in range(1, args.epochs + 1):
        codec.train(); t0 = time.time(); rm = 0.0; nb = 0
        idx = torch.randperm(N, generator=gen, device=dev)
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            x = tr[idx[i:i + args.batch_size]]
            xh, codes, commit = codec(x)
            loss = masked_mse(xh, x, mask) + commit
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
            rm += masked_mse(xh.detach(), x, mask).item(); nb += 1
        sched.step()
        if ep % max(args.epochs // 15, 1) == 0 or ep == 1:
            m = evaluate(codec, te, mask)
            score = m["ve"]
            if score > best:
                best = score
                torch.save({"model": codec.state_dict(),
                            "config": dict(vol_shape=list(fable_data.VOL_SHAPE), hidden=args.hidden,
                                           dim=args.dim, n_down=args.n_down, depth=args.depth,
                                           K=args.K, commit_beta=args.commit_beta),
                            "meta": {k: d[k] for k in ("mask", "mu", "sd", "class_names")},
                            "eval": m, "raw_bits_per_vol": raw_bpv}, args.out)
            print(f"ep{ep:3d}/{args.epochs} train_mse={rm/nb:.4f} | "
                  f"test VE={m['ve']*100:.1f}% corr={m['corr']:.3f} mse={m['mse']:.4f} "
                  f"codes={m['used']}/{args.K} ({time.time()-t0:.1f}s)")
    print(f"\nbest test VE={best*100:.1f}%  -> {args.out}")


if __name__ == "__main__":
    main()
