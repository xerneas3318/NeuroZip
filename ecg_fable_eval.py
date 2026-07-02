"""
ecg_fable_eval.py -- ECG: reconstruction fidelity + DIAGNOSIS PRESERVATION.

The paper metric: after compressing/decompressing the ECG, does the clinical
diagnosis survive? We train a small 1-D CNN to predict the PTB-XL 5-way diagnostic
superclass from the raw ECG (on ORIGINAL train records), then score it on
ORIGINAL vs DECOMPRESSED test records. If accuracy holds, the codec kept the
diagnostic content -- the NeuroZip thesis on a clinical task.

    python ecg_fable_eval.py --ckpt checkpoints/ecg_fable.pt
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ecg_fable_data as D
from ecg_codec import ECGCodec, N_LEADS

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def reconstruct_records(codec, recs):
    """(N,12,1000) -> compress/decompress each of the 4 windows, reassemble."""
    codec.eval()
    x = torch.from_numpy(recs).to(DEV)
    n = D.REC_LEN // D.WIN
    out = torch.empty_like(x)
    for w in range(n):
        seg = x[:, :, w*D.WIN:(w+1)*D.WIN]
        rec = []
        for i in range(0, len(seg), 512):
            xh, _ = codec.compress_then_reconstruct(seg[i:i+512])
            rec.append(xh)
        out[:, :, w*D.WIN:(w+1)*D.WIN] = torch.cat(rec)
    return out.cpu().numpy()


class ECGNet(nn.Module):
    """Small 1-D CNN diagnostic classifier (5-way)."""
    def __init__(self, n_leads=12, n_cls=5, h=48):
        super().__init__()
        def blk(ci, co, k=7, s=2):
            return nn.Sequential(nn.Conv1d(ci, co, k, stride=s, padding=k//2),
                                 nn.GroupNorm(8, co), nn.GELU())
        self.net = nn.Sequential(blk(n_leads, h), blk(h, h*2), blk(h*2, h*2, 5),
                                 blk(h*2, h*2, 3), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(h*2, n_cls)

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1))


def train_clf(Xtr, ytr, epochs=40, bs=256, lr=1e-3):
    torch.manual_seed(0)
    clf = ECGNet().to(DEV)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # class-balanced loss (NORM dominates)
    cnt = np.bincount(ytr, minlength=5); w = torch.tensor(cnt.sum()/(cnt+1), dtype=torch.float32, device=DEV)
    Xtr = torch.from_numpy(Xtr).to(DEV); ytr = torch.from_numpy(ytr).to(DEV)
    N = len(Xtr); g = torch.Generator(device=DEV).manual_seed(0)
    for ep in range(epochs):
        clf.train(); idx = torch.randperm(N, generator=g, device=DEV)
        for i in range(0, N-bs+1, bs):
            b = idx[i:i+bs]
            loss = F.cross_entropy(clf(Xtr[b]), ytr[b], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sched.step()
    return clf


@torch.no_grad()
def score(clf, X, y):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    clf.eval()
    logits = torch.cat([clf(torch.from_numpy(X[i:i+512]).to(DEV)) for i in range(0, len(X), 512)])
    prob = logits.softmax(1).cpu().numpy(); pred = prob.argmax(1)
    acc = (pred == y).mean()
    bacc = balanced_accuracy_score(y, pred)
    try:
        auc = roc_auc_score(np.eye(5)[y], prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return acc, bacc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ecg_fable.pt")
    args = ap.parse_args()
    d = D.load(); folds = d["folds"]
    tr_m, te_m = folds <= 8, folds == 10
    Rtr, ytr = d["recs"][tr_m], d["labels"][tr_m]
    Rte, yte = d["recs"][te_m], d["labels"][te_m]

    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    codec = ECGCodec(**ck["config"]).to(DEV); codec.load_state_dict(ck["model"]); codec.eval()

    # fidelity (windows)
    tew = D.windows(Rte); tew_t = torch.from_numpy(tew).to(DEV)
    xh = reconstruct_records(codec, Rte)
    mse = float(((xh - Rte)**2).mean()); var = float(Rte.var()); ve = 1 - mse/var
    psnr = float(10*np.log10((Rte.max()-Rte.min())**2 / mse))

    print("="*60)
    print(f"ECG (PTB-XL) fable codec | {args.ckpt}")
    print(f"  params {sum(p.numel() for p in codec.parameters())/1e6:.2f}M  "
          f"ratio {ck['eval']['ratio']:.0f}x  test records {len(Rte)}")
    print("-"*60)
    print(f"FIDELITY  MSE {mse:.4f}  VE {ve*100:.1f}%  PSNR {psnr:.1f} dB")
    print("-"*60)
    print("DIAGNOSIS PRESERVATION (5-way superclass; small 1-D CNN)")
    Rtr_d = reconstruct_records(codec, Rtr)
    clf = train_clf(Rtr, ytr)
    a_o, b_o, auc_o = score(clf, Rte, yte)
    a_a, b_a, auc_a = score(clf, xh, yte)           # archival: orig-trained -> decompressed
    clf_d = train_clf(Rtr_d, ytr)
    a_w, b_w, auc_w = score(clf_d, xh, yte)         # within: decomp-trained -> decompressed
    print(f"  chance (5-way, balanced) : 20%")
    print(f"  baseline orig->orig      : acc {a_o*100:.1f}%  bal-acc {b_o*100:.1f}%  AUC {auc_o:.3f}")
    print(f"  archival orig->decomp    : acc {a_a*100:.1f}%  bal-acc {b_a*100:.1f}%  AUC {auc_a:.3f}  "
          f"(retain {auc_a/auc_o*100:.0f}%)")
    print(f"  within   decomp->decomp  : acc {a_w*100:.1f}%  bal-acc {b_w*100:.1f}%  AUC {auc_w:.3f}  "
          f"(retain {auc_w/auc_o*100:.0f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
