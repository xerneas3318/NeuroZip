"""Group A / rate control: A5 one variable-rate model vs N fixed-rate models.

A gained codec (Cui et al. 2021 style): a single encoder/decoder/prior plus a
small learnable per-channel gain vector per rate level. Larger gain spreads the
latent over more quantization bins => higher rate, lower distortion. One model
covers all rate points; we sample the rate level each step during training.

Baseline: train a SEPARATE fixed codec at each rate level's lambda. Question:
does the single model's rate-distortion curve match the dedicated per-rate
models (so we can ship one model with a rate dial instead of four checkpoints)?

Run:  .venv/bin/python tests/exp_variable_rate.py
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import exp_common as C
from codec import quantize

LAMBDAS = [0.08, 0.02, 0.005]        # high-rate-penalty -> low; one per level


class VariableRateCodec(nn.Module):
    def __init__(self, n_levels: int):
        super().__init__()
        c = C.make_codec(**C.TINY)
        self.enc, self.dec, self.prior = c.encoder, c.decoder, c.prior
        # per-level, per-channel gain (positive via softplus); init spread so
        # level 0 is coarse (small gain) and last is fine (large gain).
        init = torch.linspace(-1.0, 1.0, n_levels)[:, None].repeat(1, C.TINY["c_lat"])
        self.raw_gain = nn.Parameter(init)
        self.n_levels = n_levels

    def gain(self, r):
        return F.softplus(self.raw_gain[r])[None, :, None] + 1e-3

    def forward(self, x, r):
        g = self.gain(r)
        y = self.enc(x) * g
        yq = quantize(y, self.training)
        bps = self.prior.bits(yq)
        xh = self.dec(yq / g)
        return xh, bps


def train_variable(data, dev, epochs=16, batch=64):
    m = VariableRateCodec(len(LAMBDAS)).to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(epochs):
        for (x,) in C.iterate(data["eeg_train"], batch=batch, shuffle=True, dev=dev):
            r = int(torch.randint(0, len(LAMBDAS), (1,)))
            xh, bps = m(x, r)
            loss = F.mse_loss(xh, x) + LAMBDAS[r] * bps
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def eval_variable(m, data, dev, r, batch=128):
    m.eval()
    g = m.gain(r)
    mse, bps, n = 0.0, 0.0, 0
    for (x,) in C.iterate(data["eeg_test"], batch=batch, shuffle=False, dev=dev):
        y = m.enc(x) * g
        yq = torch.round(y)
        xh = m.dec(yq / g)
        mse += F.mse_loss(xh, x, reduction="sum").item()
        bps += m.prior.bits(yq).item() * x.shape[0]
        n += x.shape[0]
    return mse / (n * C.N_CH * C.N_T), C.latent_bpp(bps / n, C.TINY["c_lat"])


class FixedCodec(nn.Module):
    def __init__(self):
        super().__init__()
        c = C.make_codec(**C.TINY)
        self.enc, self.dec, self.prior = c.encoder, c.decoder, c.prior

    def forward(self, x):
        y = self.enc(x); yq = quantize(y, self.training)
        return self.dec(yq), self.prior.bits(yq)


def train_fixed(data, dev, lam, epochs=16, batch=64):
    m = FixedCodec().to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(epochs):
        for (x,) in C.iterate(data["eeg_train"], batch=batch, shuffle=True, dev=dev):
            xh, bps = m(x)
            loss = F.mse_loss(xh, x) + lam * bps
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def eval_fixed(m, data, dev, batch=128):
    m.eval()
    mse, bps, n = 0.0, 0.0, 0
    for (x,) in C.iterate(data["eeg_test"], batch=batch, shuffle=False, dev=dev):
        y = m.enc(x); yq = torch.round(y)
        xh = m.dec(yq)
        mse += F.mse_loss(xh, x, reduction="sum").item()
        bps += m.prior.bits(yq).item() * x.shape[0]
        n += x.shape[0]
    return mse / (n * C.N_CH * C.N_T), C.latent_bpp(bps / n, C.TINY["c_lat"])


def main():
    C.seed_all(0)
    dev = C.device()
    C.banner(f"A5  variable-rate single model vs fixed per-rate models  (device={dev})")
    data = C.make_dataset(n_concepts=40, trials_per_concept=40, noise=0.8, seed=0)

    vr = train_variable(data, dev)
    variable_pts = []
    for r in range(len(LAMBDAS)):
        mse, bpp = eval_variable(vr, data, dev, r)
        variable_pts.append({"level": r, "lambda": LAMBDAS[r], "bpp": round(bpp, 4),
                             "mse": round(mse, 4), "ratio": round(C.ratio_vs_fp16(bpp), 1)})

    fixed_pts = []
    for lam in LAMBDAS:
        m = train_fixed(data, dev, lam)
        mse, bpp = eval_fixed(m, data, dev)
        fixed_pts.append({"lambda": lam, "bpp": round(bpp, 4), "mse": round(mse, 4),
                          "ratio": round(C.ratio_vs_fp16(bpp), 1)})

    # pair by lambda level and compare MSE
    compare = []
    for v, f in zip(variable_pts, fixed_pts):
        compare.append({"lambda": v["lambda"], "variable_bpp": v["bpp"], "fixed_bpp": f["bpp"],
                        "variable_mse": v["mse"], "fixed_mse": f["mse"],
                        "mse_gap_pct": round(100 * (v["mse"] / f["mse"] - 1), 2)})

    results = {"variable_rate_one_model": variable_pts,
               "fixed_rate_dedicated_models": fixed_pts,
               "paired_comparison": compare,
               "note": "one model produced all variable_rate points; fixed used one model each"}
    print("RESULTS_JSON:" + json.dumps(results))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
