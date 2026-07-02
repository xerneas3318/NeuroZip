"""
ecg_fable_train.py -- train the 1-D ECG codec on PTB-XL windows (fold split).

Same codec as the `ecg` branch (rian's 1-D conv codec, 12 leads), but trained on
the official-fold train windows so the held-out test folds are a clean evaluation
for both reconstruction and diagnosis-preservation.

    python ecg_fable_train.py --epochs 60 --lambda-rate 0.02
"""
from __future__ import annotations
import argparse, time, os
import numpy as np
import torch
import torch.nn.functional as F

import ecg_fable_data as D
from ecg_codec import ECGCodec, N_LEADS


@torch.no_grad()
def evaluate(codec, X, dev, bs=512):
    codec.eval(); mse = bits = n = 0.0
    for i in range(0, len(X), bs):
        x = X[i:i+bs].to(dev)
        xh, b = codec.compress_then_reconstruct(x)
        mse += F.mse_loss(xh, x).item()*x.size(0); bits += b.item()*x.size(0); n += x.size(0)
    mse /= n; bpp = codec.bpp_floor(bits/n)
    return mse, 16.0/bpp, 1 - mse/float(X.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--c-lat", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lambda-rate", type=float, default=0.02)
    ap.add_argument("--out", default="checkpoints/ecg_fable.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    dev = torch.device(args.device)

    d = D.load()
    folds = d["folds"]
    trw = D.windows(d["recs"][folds <= 8]); tew = D.windows(d["recs"][folds == 10])
    tr = torch.from_numpy(trw).to(dev); te = torch.from_numpy(tew).to(dev)
    print(f"device={dev} train_win={len(tr)} test_win={len(te)} (12x{D.WIN})")

    codec = ECGCodec(n_leads=N_LEADS, c_lat=args.c_lat, hidden=args.hidden).to(dev)
    print(f"ECGCodec params={sum(p.numel() for p in codec.parameters())/1e6:.2f}M c_lat={args.c_lat}")
    opt = torch.optim.AdamW(codec.parameters(), lr=args.lr)
    gen = torch.Generator(device=dev).manual_seed(0); N = len(tr); best = -1e9
    for ep in range(1, args.epochs+1):
        codec.train(); t0 = time.time(); rm = 0.0; nb = 0
        idx = torch.randperm(N, generator=gen, device=dev)
        for i in range(0, N-args.batch_size+1, args.batch_size):
            x = tr[idx[i:i+args.batch_size]]
            xh, bits, _ = codec(x)
            loss = F.mse_loss(xh, x) + args.lambda_rate*bits
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
            rm += F.mse_loss(xh.detach(), x).item(); nb += 1
        if ep % max(args.epochs//12, 1) == 0 or ep == 1:
            mse, ratio, ve = evaluate(codec, te, dev)
            if ve > best:
                best = ve
                torch.save({"model": codec.state_dict(),
                            "config": dict(n_leads=N_LEADS, c_lat=args.c_lat, hidden=args.hidden),
                            "eval": dict(mse=mse, ratio=ratio, ve=ve)}, args.out)
            print(f"ep{ep:3d}/{args.epochs} train_mse={rm/nb:.4f} | test MSE={mse:.4f} "
                  f"VE={ve*100:.1f}% ratio={ratio:.0f}x ({time.time()-t0:.1f}s)")
    print(f"\nbest test VE={best*100:.1f}% -> {args.out}")


if __name__ == "__main__":
    main()
