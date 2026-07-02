"""Group B (separate modality): task-aware image compression for FACE IDENTITY.

Transfers the NeuroZip idea off EEG: compress so the downstream task survives,
scored by a frozen judge in the loss. Here the modality is face images and the
task is identity.

  judge J  : a small identity embedder trained on the train split, then frozen
             (the analog of NeuroZip's frozen EEG->CLIP projector).
  codecs   : same 2D conv autoencoder + factorized prior both ways; differ only
             in lambda_task:
               fidelity    : mse + rate
               task-aware  : mse + rate + lambda_task*(1 - cos(J(x_hat), J(x)))
  eval     : identity retrieval on the TEST split (compress->decompress, embed
             with J, nearest identity centroid), plus verification ROC-AUC, at
             matched bitrate. An INDEPENDENT second embedder J2 (different seed)
             is also used at eval as a held-out judge (circularity defense).

Dataset: tries torchvision CelebA (identity target); falls back to LFW via
scikit-learn (reliable, ~200 MB, identity-labeled) if the CelebA download
fails. Both are well under 10 GB.

Run:  .venv/bin/python tests/exp_b_face.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as C

IMG = 64
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def _resize_batch(x, size=IMG):
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def load_faces(min_per_person=20, max_people=60, max_per_person=40):
    """Return dict with train/test image tensors (N,3,64,64) in [0,1] + labels.

    Tries CelebA; falls back to LFW (sklearn). Keeps identities with enough
    images so retrieval has a gallery per person.
    """
    src = None
    imgs = labels = None

    # ---- attempt 1: torchvision CelebA (identity) ----
    try:
        from torchvision.datasets import CelebA
        from torchvision import transforms
        tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor()])
        ds = CelebA(root=str(DATA_DIR), split="train", target_type="identity",
                    download=True, transform=tf)
        # subsample: pick identities with many images (light: cap counts)
        ids = np.array([int(ds.identity[i].item()) for i in range(len(ds))])
        from collections import Counter
        cnt = Counter(ids.tolist())
        keep_ids = [i for i, c in cnt.items() if c >= min_per_person][:max_people]
        keep_set = set(keep_ids)
        remap = {v: k for k, v in enumerate(keep_ids)}
        per = {i: 0 for i in keep_ids}
        xs, ys = [], []
        for i in range(len(ds)):
            idv = int(ds.identity[i].item())
            if idv in keep_set and per[idv] < max_per_person:
                x, _ = ds[i]
                xs.append(x); ys.append(remap[idv]); per[idv] += 1
            if all(v >= max_per_person for v in per.values()):
                break
        imgs = torch.stack(xs); labels = torch.tensor(ys)
        src = "CelebA"
    except Exception as e:  # noqa
        print(f"[face] CelebA unavailable ({type(e).__name__}: {str(e)[:120]}); falling back to LFW")

    # ---- attempt 2: LFW via sklearn ----
    if imgs is None:
        from sklearn.datasets import fetch_lfw_people
        lfw = fetch_lfw_people(min_faces_per_person=min_per_person, resize=0.5, color=True)
        x = torch.from_numpy(lfw.images).float()           # (N,H,W,3) in [0,255] or [0,1]
        if x.max() > 1.5:
            x = x / 255.0
        x = x.permute(0, 3, 1, 2)                           # (N,3,H,W)
        x = _resize_batch(x)
        y = torch.from_numpy(lfw.target).long()
        # cap people/images to stay light
        keep = [i for i in torch.unique(y).tolist()][:max_people]
        remap = {v: k for k, v in enumerate(keep)}
        mask = torch.tensor([int(t) in set(keep) for t in y])
        x, y = x[mask], y[mask]
        y = torch.tensor([remap[int(t)] for t in y])
        imgs, labels = x, y
        src = "LFW"

    # per-identity 70/30 split so eval identities are seen (retrieval by centroid)
    g = torch.Generator().manual_seed(0)
    tr_idx, te_idx = [], []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        k = max(1, int(0.3 * len(idx)))
        te_idx += idx[:k].tolist(); tr_idx += idx[k:].tolist()
    tr_idx = torch.tensor(tr_idx); te_idx = torch.tensor(te_idx)
    n_id = int(labels.max().item()) + 1
    print(f"[face] source={src}  images={len(imgs)}  identities={n_id}  "
          f"train={len(tr_idx)}  test={len(te_idx)}")
    return {
        "x_train": imgs[tr_idx], "y_train": labels[tr_idx],
        "x_test": imgs[te_idx], "y_test": labels[te_idx],
        "n_id": n_id, "source": src,
    }


# --------------------------------------------------------------------------- #
# Identity judge + image codec
# --------------------------------------------------------------------------- #

class IdentityEmbedder(nn.Module):
    """Small CNN -> L2-normalized 128-d identity embedding."""
    def __init__(self, dim=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),    # 32
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),   # 16
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.GELU(), # 8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Linear(128, dim)

    def forward(self, x):
        return F.normalize(self.head(self.body(x)), dim=-1)


class Factorized2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.log_b = nn.Parameter(torch.zeros(ch))

    def bits(self, y):
        b = F.softplus(self.log_b)[None, :, None, None].clamp_min(1e-4)
        cdf = lambda x: 0.5 + 0.5 * torch.sign(x) * (1 - torch.exp(-x.abs() / b))
        p = (cdf(y + 0.5) - cdf(y - 0.5)).clamp_min(1e-9)
        return -torch.log2(p).mean()


class ImageCodec(nn.Module):
    def __init__(self, c_lat=24):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 48, 5, 2, 2), nn.GroupNorm(8, 48), nn.GELU(),    # 32
            nn.Conv2d(48, 96, 5, 2, 2), nn.GroupNorm(8, 96), nn.GELU(),   # 16
            nn.Conv2d(96, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.GELU(),   # 8
            nn.Conv2d(96, c_lat, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(c_lat, 96, 1), nn.GELU(),
            nn.ConvTranspose2d(96, 96, 4, 2, 1), nn.GroupNorm(8, 96), nn.GELU(),   # 16
            nn.ConvTranspose2d(96, 48, 4, 2, 1), nn.GroupNorm(8, 48), nn.GELU(),   # 32
            nn.ConvTranspose2d(48, 3, 4, 2, 1),                                    # 64
        )
        self.prior = Factorized2D(c_lat)
        self.c_lat = c_lat

    def quant(self, y):
        if self.training:
            return y + (torch.rand_like(y) - 0.5)
        return torch.round(y)

    def forward(self, x):
        y = self.enc(x); yq = self.quant(y)
        return torch.sigmoid(self.dec(yq)), self.prior.bits(yq)

    def bpp(self, bits_per_symbol):
        # latent 8x8, image 64x64x3 pixels
        return bits_per_symbol * (self.c_lat * 8 * 8) / (IMG * IMG)


def train_judge(x, y, dev, n_id, epochs=20, batch=128, seed=0):
    torch.manual_seed(seed)
    emb = IdentityEmbedder().to(dev).train()
    head = nn.Linear(128, n_id).to(dev)
    opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=2e-3)
    for _ in range(epochs):
        for xb, yb in C.iterate(x, y, batch=batch, shuffle=True, dev=dev):
            logits = head(emb.head(emb.body(xb)))
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    emb.eval()
    for p in emb.parameters():
        p.requires_grad_(False)
    return emb


def train_codec(mode, data, dev, judge, epochs=25, batch=128, lam_rate=0.03, lam_task=3.0):
    m = ImageCodec().to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    x_all = data["x_train"]
    for _ in range(epochs):
        for (xb,) in C.iterate(x_all, batch=batch, shuffle=True, dev=dev):
            xh, bps = m(xb)
            loss = F.mse_loss(xh, xb) + lam_rate * bps
            if mode == "task":
                with torch.no_grad():
                    tgt = judge(xb)
                emb = judge(xh)
                loss = loss + lam_task * (1 - F.cosine_similarity(emb, tgt, dim=-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def evaluate(m, data, dev, judge, judge2):
    m.eval()
    # gallery centroids from TRAIN embeddings of decompressed images
    def embed_set(x, J):
        outs = []
        for (xb,) in C.iterate(x, batch=128, shuffle=False, dev=dev):
            y = m.enc(xb); xh = torch.sigmoid(m.dec(torch.round(y)))
            outs.append(J(xh).cpu())
        return torch.cat(outs, 0)

    def metrics(J):
        tr = embed_set(data["x_train"], J)
        te = embed_set(data["x_test"], J)
        n_id = data["n_id"]
        cent = torch.zeros(n_id, tr.shape[1])
        for c in range(n_id):
            mask = data["y_train"] == c
            if mask.any():
                cent[c] = F.normalize(tr[mask].mean(0), dim=-1)
        sims = F.normalize(te, dim=-1) @ cent.t()
        pred = sims.argmax(1)
        top1 = (pred == data["y_test"]).float().mean().item()
        top5 = (sims.argsort(1, descending=True)[:, :5]
                == data["y_test"].unsqueeze(1)).any(1).float().mean().item()
        # verification AUC over test pairs
        te_n = F.normalize(te, dim=-1)
        P = min(2000, len(te))
        gi = torch.Generator().manual_seed(1)
        a = torch.randint(0, len(te), (P,), generator=gi)
        b = torch.randint(0, len(te), (P,), generator=gi)
        cos = (te_n[a] * te_n[b]).sum(1)
        same = (data["y_test"][a] == data["y_test"][b]).float()
        auc = _auc(cos.numpy(), same.numpy())
        return {"id_top1": round(top1, 4), "id_top5": round(top5, 4), "verif_auc": round(auc, 4)}

    # rate + fidelity
    bps, mse, n = 0.0, 0.0, 0
    for (xb,) in C.iterate(data["x_test"], batch=128, shuffle=False, dev=dev):
        y = m.enc(xb); yq = torch.round(y); xh = torch.sigmoid(m.dec(yq))
        bps += m.prior.bits(yq).item() * xb.shape[0]
        mse += F.mse_loss(xh, xb, reduction="sum").item() / (3 * IMG * IMG)
        n += xb.shape[0]
    bpp = m.bpp(bps / n)
    out = {"bpp": round(bpp, 4), "ratio_vs_uint8_rgb": round(24.0 / max(bpp, 1e-6), 1),
           "mse": round(mse / n, 5),
           "judge_same": metrics(judge), "judge_independent": metrics(judge2)}
    return out


def _auc(scores, labels):
    """ROC-AUC via the rank (Mann-Whitney U) form; no numpy.trapz dependency."""
    P = labels.sum(); N = len(labels) - P
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")     # ascending
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - P * (P + 1) / 2) / (P * N))


def _interp(xs, ys, x):
    pts = sorted(zip(xs, ys))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            f = (x - xs[i]) / (xs[i + 1] - xs[i] + 1e-9)
            return ys[i] + f * (ys[i + 1] - ys[i])
    return ys[-1]


def main():
    C.seed_all(0)
    dev = C.device()
    C.banner(f"B  task-aware FACE codec vs fidelity (matched bitrate)  (device={dev})")
    data = load_faces()
    judge = train_judge(data["x_train"], data["y_train"], dev, data["n_id"], seed=0)
    judge2 = train_judge(data["x_train"], data["y_train"], dev, data["n_id"], seed=7)  # held-out

    # Sweep lambda_rate per mode so the rate-vs-accuracy curves overlap; the task
    # loss inflates rate, so it needs higher lambda to reach the same bpp.
    sweeps = {"fidelity": (0.03, 0.1, 0.3, 0.8), "task": (0.1, 0.3, 0.8, 2.0)}
    curves = {}
    for mode, lams in sweeps.items():
        pts = []
        for lam in lams:
            m = train_codec(mode, data, dev, judge, lam_rate=lam)
            ev = evaluate(m, data, dev, judge, judge2)
            pts.append({"lambda_rate": lam, "bpp": ev["bpp"], "ratio": ev["ratio_vs_uint8_rgb"],
                        "mse": ev["mse"], "id_top1": ev["judge_same"]["id_top1"],
                        "id_top5": ev["judge_same"]["id_top5"],
                        "auc": ev["judge_same"]["verif_auc"],
                        "id_top1_indep": ev["judge_independent"]["id_top1"]})
        curves[mode] = pts

    # Matched-bpp read: at each fidelity bpp within the task curve's range,
    # interpolate the task-aware id_top1.
    fb = [p["bpp"] for p in curves["fidelity"]]
    tb = [p["bpp"] for p in curves["task"]]
    t_top1 = [p["id_top1"] for p in curves["task"]]
    t_top1i = [p["id_top1_indep"] for p in curves["task"]]
    matched = []
    for p in curves["fidelity"]:
        if min(tb) <= p["bpp"] <= max(tb):
            matched.append({
                "bpp": p["bpp"], "ratio": p["ratio"],
                "fidelity_id_top1": p["id_top1"],
                "task_id_top1_at_same_bpp": round(_interp(tb, t_top1, p["bpp"]), 4),
                "delta_pp": round(100 * (_interp(tb, t_top1, p["bpp"]) - p["id_top1"]), 2),
                "fidelity_id_top1_indep": p["id_top1_indep"],
                "task_id_top1_indep_at_same_bpp": round(_interp(tb, t_top1i, p["bpp"]), 4),
            })

    results = {
        "dataset": data["source"], "n_identities": data["n_id"],
        "chance_top1": round(1 / data["n_id"], 4),
        "fidelity_curve": curves["fidelity"],
        "task_aware_curve": curves["task"],
        "matched_bitrate": matched,
    }
    print("RESULTS_JSON:" + json.dumps(results))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
