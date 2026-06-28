"""
fmri_neurozip.py — NeuroZip on fMRI (THINGS-data, IT cortex).

The EEG thesis, transferred to fMRI: a TASK-AWARE codec (reconstruction + a CLIP
loss routed through a frozen fMRI->CLIP judge) preserves semantic retrievability
under compression better than a FIDELITY-only codec at the same compression.

Data: THINGS-fMRI single-trial responses to the 100 test images, restricted to
IT cortex (the object-semantic region). 100 images x 12 reps per subject.

Honest protocol (no train/eval leakage):
  reps 1-6  -> fit the frozen judge W (ridge fMRI->CLIP) AND train the codecs
  reps 7-12 -> averaged to one pattern per image; the held-out eval set
Retrieval gallery = the 100 concepts' CLIP text features.

Run:  python fmri_neurozip.py --subject sub-01 --latent 64
"""
from __future__ import annotations
import argparse, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = Path("data_fmri/betas_csv_testset")
EEGFEAT = "data/Preprocessed_data_250Hz_whiten/ViT-B-32_features_{}.pt"


def concept(f): return re.sub(r"_\d+[a-z]+\.jpg$", "", f)


def load(subject):
    tr = torch.load(EEGFEAT.format("train"), weights_only=False)
    te = torch.load(EEGFEAT.format("test"), weights_only=False)
    TXT = {str(k): v.float().numpy() for k, v in {**tr["text_features"], **te["text_features"]}.items()}
    B = np.load(DATA / f"{subject}_TestResponsesIT.npy").astype(np.float32)   # (1200, V)
    m = pd.read_csv(DATA / f"{subject}_StimulusMetadataTestset.csv")
    imgs = m["stimulus"].values
    uniq = sorted(set(imgs))
    half1, half2 = [], []
    for u in uniq:
        r = B[imgs == u]
        half1.append(r[:6].mean(0)); half2.append(r[6:].mean(0))           # 6-rep averages
    X1 = np.stack(half1); X2 = np.stack(half2)                              # (100, V) each
    Y = np.stack([TXT[concept(u)] for u in uniq])                          # (100, 512) CLIP gallery
    return X1, X2, Y, uniq


def retrieval(emb, gallery, ks=(1, 5, 10)):
    emb = F.normalize(emb, dim=1); gallery = F.normalize(gallery, dim=1)
    order = (emb @ gallery.t()).argsort(1, descending=True)
    gold = torch.arange(emb.size(0), device=emb.device)
    return {f"top{k}": (order[:, :k] == gold[:, None]).any(1).float().mean().item() for k in ks}


class Codec(nn.Module):
    """MLP autoencoder over the IT voxel vector. Latent dim = compression knob."""
    def __init__(self, V, latent=64, hidden=512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(V, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, V))
        self.V, self.latent = V, latent

    def forward(self, x):
        z = self.enc(x)
        z = z + (torch.rand_like(z) - 0.5) if self.training else torch.round(z)  # scalar quant
        return self.dec(z), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="sub-01")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lam_task", type=float, default=10.0)
    ap.add_argument("--judge_lam", type=float, default=2e3)
    args = ap.parse_args()

    X1, X2, Y, uniq = load(args.subject)
    V = X1.shape[1]
    mu, sd = X1.mean(0), X1.std(0) + 1e-6                                   # standardize on train half
    x1 = torch.tensor((X1 - mu) / sd, device=DEV)
    x2 = torch.tensor((X2 - mu) / sd, device=DEV)
    Yt = torch.tensor(Y, device=DEV)
    gallery = Yt.clone()

    # ---- frozen judge W: ridge fMRI(train half) -> CLIP ----
    W = torch.linalg.solve(x1.t() @ x1 + args.judge_lam * torch.eye(V, device=DEV), x1.t() @ Yt)
    def judge(x): return x @ W                                             # x already standardized
    base = retrieval(judge(x2), gallery)
    print(f"[{args.subject}] V={V}  frozen judge on held-out reps: "
          f"top1={base['top1']*100:.1f} top5={base['top5']*100:.1f} top10={base['top10']*100:.1f}  "
          f"(chance 1/5/10%)")

    def train_codec(task_aware):
        torch.manual_seed(0)
        codec = Codec(V, latent=args.latent).to(DEV)
        opt = torch.optim.Adam(codec.parameters(), lr=1e-3, weight_decay=1e-4)
        Ytn = F.normalize(Yt, dim=1)
        for ep in range(args.epochs):
            codec.train(); opt.zero_grad()
            xh, _ = codec(x1)
            mse = F.mse_loss(xh, x1)
            loss = mse
            if task_aware:
                emb = judge(xh)                                            # gradient flows through frozen W
                loss = loss + args.lam_task * (1 - F.cosine_similarity(emb, Ytn, dim=1).mean())
            loss.backward(); opt.step()
        codec.eval()
        with torch.no_grad():
            xh, _ = codec(x2)
            mse = F.mse_loss(xh, x2).item()
            r = retrieval(judge(xh), gallery)
        return mse, r

    fid_mse, fid_r = train_codec(False)
    tsk_mse, tsk_r = train_codec(True)
    ratio = V / args.latent
    print(f"\n  compression ~{ratio:.0f}x  (V={V} -> latent={args.latent})")
    print(f"  {'codec':16s} {'MSE':>8} {'top1':>7} {'top5':>7} {'top10':>7}")
    print(f"  {'fidelity-only':16s} {fid_mse:8.4f} {fid_r['top1']*100:6.1f} {fid_r['top5']*100:6.1f} {fid_r['top10']*100:6.1f}")
    print(f"  {'task-aware':16s} {tsk_mse:8.4f} {tsk_r['top1']*100:6.1f} {tsk_r['top5']*100:6.1f} {tsk_r['top10']*100:6.1f}")
    print(f"  top5 lift (task - fidelity): {(tsk_r['top5']-fid_r['top5'])*100:+.1f} pp")


if __name__ == "__main__":
    main()
