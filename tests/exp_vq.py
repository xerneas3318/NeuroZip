"""Group A / quantizer: A4 VQ bottleneck vs scalar-uniform-noise quantization.

Same encoder/decoder architecture (from codec.py) both ways; only the
bottleneck changes:

* scalar  -- the shipped path: round + factorized-Laplace rate.
* VQ      -- vector-quantize the 32 temporal tokens (dim c_lat) to a learned
             codebook of size K; rate = empirical entropy of code usage (what
             an entropy coder over the indices would actually pay).

We trace a small rate-distortion curve for each (scalar over lambda_rate, VQ
over K) and report MSE at matched bitrate. Question: does VQ buy lower
distortion per bit at the low bitrates NeuroZip targets?

Run:  .venv/bin/python tests/exp_vq.py
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import exp_common as C


class VQBottleneck(nn.Module):
    """VQ-VAE bottleneck over temporal tokens with straight-through + losses."""
    def __init__(self, dim: int, k: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(k, dim) * 0.1)
        self.k = k
        self.beta = beta

    def forward(self, y):                      # y: (B, dim, T)
        b, d, t = y.shape
        tok = y.transpose(1, 2).reshape(-1, d)               # (B*T, dim)
        dist = (tok.pow(2).sum(1, keepdim=True)
                - 2 * tok @ self.codebook.t()
                + self.codebook.pow(2).sum(1))
        idx = dist.argmin(1)                                 # (B*T,)
        q = self.codebook[idx]
        vq_loss = F.mse_loss(q, tok.detach()) + self.beta * F.mse_loss(q.detach(), tok)
        q_st = tok + (q - tok).detach()                      # straight-through
        q_st = q_st.reshape(b, t, d).transpose(1, 2)
        return q_st, idx.reshape(b, t), vq_loss

    def code_entropy_bits(self, idx):          # empirical entropy of indices
        counts = torch.bincount(idx.reshape(-1), minlength=self.k).float()
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * torch.log2(p)).sum())


class ScalarCodec(nn.Module):
    def __init__(self):
        super().__init__()
        c = C.make_codec(**C.TINY)
        self.enc, self.dec, self.prior = c.encoder, c.decoder, c.prior

    def forward(self, x):
        from codec import quantize
        y = self.enc(x)
        yq = quantize(y, self.training)
        bps = self.prior.bits(yq)
        return self.dec(yq), bps


class VQCodec(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        c = C.make_codec(**C.TINY)
        self.enc, self.dec = c.encoder, c.decoder
        self.vq = VQBottleneck(dim=C.TINY["c_lat"], k=k)

    def forward(self, x):
        y = self.enc(x)
        q, idx, vq_loss = self.vq(y)
        return self.dec(q), idx, vq_loss


def train_scalar(data, dev, lam_rate, epochs=14, batch=64):
    m = ScalarCodec().to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(epochs):
        for (x,) in C.iterate(data["eeg_train"], batch=batch, shuffle=True, dev=dev):
            xh, bps = m(x)
            loss = F.mse_loss(xh, x) + lam_rate * bps
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def eval_scalar(m, data, dev, batch=128):
    m.eval()
    mse, bps, n = 0.0, 0.0, 0
    for (x,) in C.iterate(data["eeg_test"], batch=batch, shuffle=False, dev=dev):
        y = m.enc(x); yq = torch.round(y)
        xh = m.dec(yq)
        mse += F.mse_loss(xh, x, reduction="sum").item()
        bps += m.prior.bits(yq).item() * x.shape[0]
        n += x.shape[0]
    bpsym = bps / n
    return mse / (n * C.N_CH * C.N_T), C.latent_bpp(bpsym, C.TINY["c_lat"])


def train_vq(data, dev, k, epochs=14, batch=64):
    m = VQCodec(k).to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(epochs):
        for (x,) in C.iterate(data["eeg_train"], batch=batch, shuffle=True, dev=dev):
            xh, idx, vq_loss = m(x)
            loss = F.mse_loss(xh, x) + vq_loss
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def eval_vq(m, data, dev, batch=128):
    m.eval()
    mse, n, all_idx = 0.0, 0, []
    for (x,) in C.iterate(data["eeg_test"], batch=batch, shuffle=False, dev=dev):
        xh, idx, _ = m(x)
        mse += F.mse_loss(xh, x, reduction="sum").item()
        all_idx.append(idx.cpu()); n += x.shape[0]
    idx = torch.cat(all_idx, 0)
    ent = m.vq.code_entropy_bits(idx)                 # bits per token
    t_lat = idx.shape[1]
    bpp = ent * t_lat / (C.N_CH * C.N_T)              # bits per EEG sample
    return mse / (n * C.N_CH * C.N_T), bpp, ent


def main():
    C.seed_all(0)
    dev = C.device()
    C.banner(f"A4  VQ vs scalar quantization  (device={dev})")
    data = C.make_dataset(n_concepts=40, trials_per_concept=40, noise=0.8, seed=0)

    scalar_pts, vq_pts = [], []
    for lam in (0.005, 0.02, 0.08):
        m = train_scalar(data, dev, lam)
        mse, bpp = eval_scalar(m, data, dev)
        scalar_pts.append({"lambda_rate": lam, "bpp": round(bpp, 4),
                           "mse": round(mse, 4), "ratio": round(C.ratio_vs_fp16(bpp), 1)})
    for k in (16, 64, 256):
        m = train_vq(data, dev, k)
        mse, bpp, ent = eval_vq(m, data, dev)
        vq_pts.append({"K": k, "code_entropy_bits": round(ent, 3), "bpp": round(bpp, 4),
                       "mse": round(mse, 4), "ratio": round(C.ratio_vs_fp16(bpp), 1)})

    # crude matched-rate read: for each VQ point, linear-interp scalar MSE at same bpp
    def interp_scalar_mse(bpp):
        pts = sorted(scalar_pts, key=lambda p: p["bpp"])
        xs = [p["bpp"] for p in pts]; ys = [p["mse"] for p in pts]
        if bpp <= xs[0]:
            return ys[0]
        if bpp >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= bpp <= xs[i + 1]:
                f = (bpp - xs[i]) / (xs[i + 1] - xs[i])
                return ys[i] + f * (ys[i + 1] - ys[i])
        return ys[-1]

    matched = [{"K": p["K"], "bpp": p["bpp"], "vq_mse": p["mse"],
                "scalar_mse_at_same_bpp": round(interp_scalar_mse(p["bpp"]), 4),
                "vq_mse_delta_pct": round(100 * (p["mse"] / interp_scalar_mse(p["bpp"]) - 1), 2)}
               for p in vq_pts]

    results = {"scalar_rd": scalar_pts, "vq_rd": vq_pts, "matched_rate": matched}
    print("RESULTS_JSON:" + json.dumps(results))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
