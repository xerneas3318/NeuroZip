"""
har_fable.py -- train the 1-D codec on UCI HAR, then test whether ACTIVITY
RECOGNITION survives compression (the task-preservation metric on a non-neural,
wearable-sensor modality).

    python har_fable.py
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import har_fable_data as D
from har_codec import HARCodec, N_CH

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train_codec(tr, te, epochs=80, lam=0.02, lr=3e-4, c_lat=16):
    codec = HARCodec(c_lat=c_lat).to(DEV)
    opt = torch.optim.AdamW(codec.parameters(), lr=lr)
    g = torch.Generator(device=DEV).manual_seed(0); N = len(tr); best = None; bestve = -1e9
    for ep in range(1, epochs + 1):
        codec.train(); idx = torch.randperm(N, generator=g, device=DEV)
        for i in range(0, N - 256 + 1, 256):
            x = tr[idx[i:i+256]]
            xh, bits, _ = codec(x)
            loss = F.mse_loss(xh, x) + lam * bits
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
        if ep % 20 == 0 or ep == 1:
            codec.eval()
            with torch.no_grad():
                xh, b = codec.compress_then_reconstruct(te)
                mse = F.mse_loss(xh, te).item(); ve = 1 - mse / te.var().item()
                ratio = 16.0 / codec.bpp_floor(b.item())
            if ve > bestve:
                bestve = ve; best = {k: v.clone() for k, v in codec.state_dict().items()}
            print(f"  codec ep{ep:3d} MSE={mse:.4f} VE={ve*100:.1f}% ratio={ratio:.0f}x")
    codec.load_state_dict(best); codec.eval()
    return codec, bestve, ratio


class HARNet(nn.Module):
    def __init__(self, n_ch=9, n_cls=6, h=32):
        super().__init__()
        def blk(ci, co, s=2):
            return nn.Sequential(nn.Conv1d(ci, co, 5, stride=s, padding=2),
                                 nn.GroupNorm(8, co), nn.GELU())
        self.net = nn.Sequential(blk(n_ch, h), blk(h, h*2), blk(h*2, h*2),
                                 nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(h*2, n_cls)

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1))


def train_clf(X, y, epochs=40):
    torch.manual_seed(0); clf = HARNet().to(DEV)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    N = len(X); g = torch.Generator(device=DEV).manual_seed(0)
    for ep in range(epochs):
        clf.train(); idx = torch.randperm(N, generator=g, device=DEV)
        for i in range(0, N-256+1, 256):
            b = idx[i:i+256]
            loss = F.cross_entropy(clf(X[b]), y[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    clf.eval(); return clf


@torch.no_grad()
def acc(clf, X, y):
    p = torch.cat([clf(X[i:i+512]).argmax(1) for i in range(0, len(X), 512)])
    return (p == y).float().mean().item()


def main():
    d = D.load()
    tr = torch.from_numpy(d["Xtr"]).to(DEV); te = torch.from_numpy(d["Xte"]).to(DEV)
    ytr = torch.from_numpy(d["ytr"]).to(DEV); yte = torch.from_numpy(d["yte"]).to(DEV)
    print(f"HAR: train {tuple(tr.shape)} test {tuple(te.shape)}  device {DEV}")
    print("training codec ...")
    codec, ve, ratio = train_codec(tr, te)

    with torch.no_grad():
        te_d = codec.compress_then_reconstruct(te)[0]
        tr_d = codec.compress_then_reconstruct(tr)[0]
    mse = F.mse_loss(te_d, te).item()
    psnr = 10*np.log10((te.max()-te.min()).item()**2/mse)

    print("\ntraining activity classifier ...")
    clf = train_clf(tr, ytr)
    clf_d = train_clf(tr_d, ytr)
    base = acc(clf, te, yte); arch = acc(clf, te_d, yte); within = acc(clf_d, te_d, yte)

    print("\n" + "="*56)
    print("HAR (UCI) fable codec -- activity recognition survives compression?")
    print("="*56)
    print(f"  codec: {sum(p.numel() for p in codec.parameters())/1e6:.2f}M params  "
          f"{ratio:.0f}x  MSE {mse:.4f}  VE {ve*100:.1f}%  PSNR {psnr:.1f} dB")
    print(f"  activity recognition (6-way, chance 17%):")
    print(f"    baseline orig->orig     : {base*100:.1f}%")
    print(f"    archival orig->decomp   : {arch*100:.1f}%  (retain {arch/base*100:.0f}%)")
    print(f"    within   decomp->decomp : {within*100:.1f}%  (retain {within/base*100:.0f}%)")
    print("="*56)


if __name__ == "__main__":
    main()
