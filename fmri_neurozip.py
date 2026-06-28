"""
fmri_neurozip.py — NeuroZip on fMRI (THINGS-data, IT cortex).

Trains two codecs on IT-cortex single-trial responses and saves everything the
semantic-search demo needs:
  - fidelity codec : reconstruction (MSE) only           -> does NOT see CLIP
  - actual  codec  : MSE + a CLIP loss backpropagated through a FROZEN
                     fMRI->CLIP judge                     -> knows what the fMRI is *for*

Both compress an IT pattern to the same tiny latent (same bitrate). The "actual"
codec preserves text-searchability; the fidelity one throws it away.

Data: THINGS-fMRI test stimuli (100 images x 12 reps) in IT. Honest rep-split:
reps 1-6 train the judge + codecs (600 trials); reps 7-12 averaged = held-out eval.

Run:  python fmri_neurozip.py --subject sub-01 --latent 64
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = Path("data_fmri/betas_csv_testset")
FEAT = "data/Preprocessed_data_250Hz_whiten/ViT-B-32_features_{}.pt"
OUT = Path("data_fmri/artifacts")


def concept(f): return re.sub(r"_\d+[a-z]+\.jpg$", "", f)


class Codec(nn.Module):
    def __init__(self, V, latent=64, hidden=512):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(V, hidden), nn.GELU(), nn.Linear(hidden, latent))
        self.d = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, V))

    def forward(self, x):
        z = self.e(x)
        z = z + (torch.rand_like(z) - 0.5) if self.training else torch.round(z)  # scalar quant
        return self.d(z)


def retrieval(emb, gallery, ks=(1, 5, 10)):
    e = F.normalize(emb, dim=1); g = F.normalize(gallery, dim=1)
    order = (e @ g.t()).argsort(1, descending=True)
    gold = torch.arange(emb.size(0), device=emb.device)
    return {f"top{k}": (order[:, :k] == gold[:, None]).any(1).float().mean().item() for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="sub-01")
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lam_task", type=float, default=3.0)
    ap.add_argument("--judge_lam", type=float, default=2e3)
    args = ap.parse_args()

    tr = torch.load(FEAT.format("train"), weights_only=False)
    te = torch.load(FEAT.format("test"), weights_only=False)
    TXT = {str(k): v.float().numpy() for k, v in {**tr["text_features"], **te["text_features"]}.items()}

    B = np.load(DATA / f"{args.subject}_TestResponsesIT.npy").astype(np.float32)
    m = pd.read_csv(DATA / f"{args.subject}_StimulusMetadataTestset.csv")
    imgs = m["stimulus"].values; uniq = sorted(set(imgs))
    concepts = [concept(u) for u in uniq]
    h1, h2, Y, trials, trials_y = [], [], [], [], []
    for u in uniq:
        r = B[imgs == u]
        h1.append(r[:6].mean(0)); h2.append(r[6:].mean(0)); Y.append(TXT[concept(u)])
        for t in range(6):
            trials.append(r[t]); trials_y.append(TXT[concept(u)])
    h1, h2, Y = np.stack(h1), np.stack(h2), np.stack(Y)
    V = h1.shape[1]
    mu, sd = h1.mean(0), h1.std(0) + 1e-6
    nrm = lambda X: (X - mu) / sd
    x1 = torch.tensor(nrm(h1), device=DEV); x2 = torch.tensor(nrm(h2), device=DEV)
    Xt = torch.tensor(nrm(np.stack(trials)), device=DEV); Yt = torch.tensor(np.stack(trials_y), device=DEV)
    G = torch.tensor(Y, device=DEV)

    # frozen judge: ridge fMRI(train half) -> CLIP
    W = torch.linalg.solve(x1.t() @ x1 + args.judge_lam * torch.eye(V, device=DEV), x1.t() @ G)
    judge = lambda x: x @ W
    base = retrieval(judge(x2), G)
    print(f"[{args.subject}] V={V} | frozen judge (held-out reps): "
          f"top1={base['top1']*100:.0f} top5={base['top5']*100:.0f} top10={base['top10']*100:.0f}")

    def train(task):
        torch.manual_seed(0)
        c = Codec(V, args.latent).to(DEV)
        opt = torch.optim.Adam(c.parameters(), 1e-3, weight_decay=1e-4)
        Yn = F.normalize(Yt, dim=1)
        for _ in range(args.epochs):
            c.train(); opt.zero_grad(); xh = c(Xt); loss = F.mse_loss(xh, Xt)
            if task:
                loss = loss + args.lam_task * (1 - F.cosine_similarity(judge(xh), Yn, dim=1).mean())
            loss.backward(); opt.step()
        c.eval()
        with torch.no_grad():
            xh = c(x2); return c, F.mse_loss(xh, x2).item(), retrieval(judge(xh), G), xh

    fc, fm, fr, fx = train(False)
    tc, tm, tr_, tx = train(True)
    print(f"  compression ~{V//args.latent}x")
    print(f"  fidelity : mse={fm:.3f} top1={fr['top1']*100:.0f} top5={fr['top5']*100:.0f} top10={fr['top10']*100:.0f}")
    print(f"  actual   : mse={tm:.3f} top1={tr_['top1']*100:.0f} top5={tr_['top5']*100:.0f} top10={tr_['top10']*100:.0f}"
          f"   (top5 lift {(tr_['top5']-fr['top5'])*100:+.0f}pp)")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "subject": args.subject, "V": V, "latent": args.latent,
        "concepts": concepts, "stimuli": uniq,
        "mu": mu, "sd": sd, "W": W.cpu(),
        "gallery": G.cpu(),                                  # (100,512) per-concept CLIP
        "x2": x2.cpu(),                                      # held-out eval patterns
        "fidelity": fc.cpu().state_dict(), "actual": tc.cpu().state_dict(),
        "metrics": {"judge": base, "fidelity": fr, "actual": tr_, "fid_mse": fm, "act_mse": tm,
                    "ratio": V // args.latent},
    }, OUT / f"{args.subject}.pt")
    print(f"  saved {OUT/(args.subject+'.pt')}")


if __name__ == "__main__":
    main()
