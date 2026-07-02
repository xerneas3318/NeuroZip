"""Stacked codec vs current NeuroZip (origin/main) -- head-to-head benchmark.

Combines the compatible A-series wins into ONE codec and benchmarks it against a
faithful reproduction of the shipped codec (the imported, byte-identical-to-
origin/main ``codec.EEGCodec`` + factorized Laplace prior + savez serialization
+ cosine task loss, one model per tier).

Stacked = same conv encoder/decoder as origin/main, plus:
  A1  real range coding driven by the model's own prior (actual on-disk bytes)
  A2+A3  a mixture mean-scale HYPERPRIOR (per-position K-Laplace conditional
         predicted from a hyper-latent) -- combines hyperprior + mixture
  A5  variable-rate GAINS: one model spans all tiers via a per-level gain vector
  A6  cosine task loss (the contrastive variant lost in exp_contrastive, so the
      stack keeps the simpler cosine term)
  (A4 VQ is intentionally excluded: architecturally exclusive with the scalar +
   hyperprior path, and only won in an extrapolated ultra-low-rate regime.)

Both models train under an identical budget on the same data + share the same
frozen projector. We report, at matched distortion/retrieval:
  * real ON-DISK compression ratio (savez vs range-coded) -- the headline
  * modeled bpp, MSE, retrieval top-1/5
  * #models needed (baseline: one per tier; stacked: one total)

Run:  .venv/bin/python tests/exp_stacked.py
"""

from __future__ import annotations

import io
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import exp_common as C
from codec import EEGCodec, quantize
from exp_priors import _encode, _decode, _laplace_cdf, range_code_latents

TAU = 0.07
LAM_TASK = 3.0


# --------------------------------------------------------------------------- #
# General range coder over per-symbol pmf tables (A1, mixture-capable)
# --------------------------------------------------------------------------- #

def code_and_measure(symbols_int, pmf_matrix, values, precision_bits=16):
    """Range-code integer symbols under per-symbol pmfs; verify round-trip.

    symbols_int: (M,) int; pmf_matrix: (M, V) each row a pmf over `values` (V,).
    Returns dict(actual_bits, theoretical_bits, roundtrip_ok).
    """
    total = 1 << precision_bits
    vmin = int(values[0])
    freqs = np.maximum((pmf_matrix / pmf_matrix.sum(1, keepdims=True) * total).astype(np.int64), 1)
    # fix each row to sum to `total`
    diff = total - freqs.sum(1)
    freqs[np.arange(len(freqs)), freqs.argmax(1)] += diff
    cum = np.concatenate([np.zeros((len(freqs), 1), np.int64), np.cumsum(freqs, 1)], 1)

    sym_cum, sym_freq, sym_tot, dec_tables, theo = [], [], [], [], 0.0
    for m, v in enumerate(symbols_int):
        idx = int(v - vmin)
        sym_cum.append(int(cum[m, idx])); sym_freq.append(int(freqs[m, idx]))
        sym_tot.append(total)
        dec_tables.append((values, cum[m], total))
        theo += -math.log2(freqs[m, idx] / total)
    data, nbits = _encode(sym_cum, sym_freq, sym_tot)
    decoded = _decode(data, nbits, len(symbols_int), dec_tables)
    return {"actual_bits": len(data) * 8,
            "theoretical_bits": theo,
            "roundtrip_ok": bool(decoded == [int(v) for v in symbols_int])}


# --------------------------------------------------------------------------- #
# Stacked codec: gains + mixture mean-scale hyperprior + cosine task
# --------------------------------------------------------------------------- #

def _laplace_cdf_t(x, mu, b):
    z = x - mu
    return 0.5 + 0.5 * torch.sign(z) * (1 - torch.exp(-z.abs() / b))


class _FactorizedScale(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.log_b = nn.Parameter(torch.zeros(ch))

    def bits(self, z):
        b = F.softplus(self.log_b)[None, :, None].clamp_min(1e-4)
        cdf = lambda x: 0.5 + 0.5 * torch.sign(x) * (1 - torch.exp(-x.abs() / b))
        p = (cdf(z + 0.5) - cdf(z - 0.5)).clamp_min(1e-9)
        return -torch.log2(p)

    def pmf_np(self, values):
        b = F.softplus(self.log_b).detach().cpu().numpy()          # (ch,)
        cdf = lambda x, bb: 0.5 + 0.5 * np.sign(x) * (1 - np.exp(-np.abs(x) / bb))
        out = np.stack([cdf(values + 0.5, bb) - cdf(values - 0.5, bb) for bb in b])
        return np.clip(out, 1e-12, None)                            # (ch, V)


class StackedCodec(nn.Module):
    def __init__(self, n_levels, c_lat=16, hidden=32, hyper_ch=12, K=3):
        super().__init__()
        base = EEGCodec(c_lat=c_lat, hidden=hidden, n_attn=0)       # origin/main arch
        self.enc, self.dec = base.encoder, base.decoder
        self.K, self.c_lat = K, c_lat
        # A5 gains
        init = torch.linspace(-1.0, 1.0, n_levels)[:, None].repeat(1, c_lat)
        self.raw_gain = nn.Parameter(init)
        # A2+A3 mixture mean-scale hyperprior
        self.h_enc = nn.Sequential(
            nn.Conv1d(c_lat, hyper_ch, 3, 2, 1), nn.GELU(),
            nn.Conv1d(hyper_ch, hyper_ch, 3, 2, 1))
        self.h_dec = nn.Sequential(
            nn.ConvTranspose1d(hyper_ch, hyper_ch, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose1d(hyper_ch, 3 * K * c_lat, 4, 2, 1))
        self.zprior = _FactorizedScale(hyper_ch)

    def gain(self, level):
        return F.softplus(self.raw_gain[level])[None, :, None] + 1e-3

    def _params(self, z_q, T):
        out = self.h_dec(z_q)[..., :T]                              # (B, 3*K*c_lat, T)
        B = out.shape[0]
        out = out.view(B, 3, self.K, self.c_lat, T)
        w = torch.softmax(out[:, 0], dim=1)                         # (B,K,c_lat,T)
        mu = out[:, 1]
        b = F.softplus(out[:, 2]).clamp_min(1e-3)
        return w, mu, b

    def _mix_bits(self, y_q, w, mu, b):
        yv = y_q.unsqueeze(1)                                       # (B,1,c_lat,T)
        p = (_laplace_cdf_t(yv + 0.5, mu, b) - _laplace_cdf_t(yv - 0.5, mu, b))
        pmf = (w * p).sum(1).clamp_min(1e-9)                        # (B,c_lat,T)
        return -torch.log2(pmf)

    def forward(self, x, level):
        g = self.gain(level)
        y = self.enc(x) * g
        y_q = quantize(y, self.training)
        z = self.h_enc(y)
        z_q = quantize(z, self.training)
        w, mu, b = self._params(z_q, y.shape[-1])
        bits_y = self._mix_bits(y_q, w, mu, b)                      # (B,c_lat,T)
        bits_z = self.zprior.bits(z_q)                              # (B,hyper_ch,Tz)
        xh = self.dec(y_q / g)
        return xh, bits_y, bits_z

    def total_bpp(self, bits_y, bits_z, B):
        return (bits_y.sum() + bits_z.sum()) / (B * C.N_CH * C.N_T)


# --------------------------------------------------------------------------- #
# Shared training pieces
# --------------------------------------------------------------------------- #

def train_projector(data, dev, epochs=25, batch=64):
    proj = C.make_projector(hidden=32).to(dev).train()
    proto = F.normalize(data["prototypes"].to(dev), dim=-1)
    opt = torch.optim.Adam(proj.parameters(), lr=2e-3)
    for _ in range(epochs):
        for x, y in C.iterate(data["eeg_train"], data["lab_train"], batch=batch, shuffle=True, dev=dev):
            loss = F.cross_entropy(proj(x) @ proto.t() / TAU, y)
            opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


def _cos_task(proj, xh, tgt):
    return (1 - F.cosine_similarity(proj(xh), F.normalize(tgt, dim=-1), dim=-1)).mean()


# ---- baseline: faithful origin/main codec (factorized prior, per-tier) ----

class BaselineCodec(nn.Module):
    def __init__(self, c_lat=16, hidden=32):
        super().__init__()
        self.m = EEGCodec(c_lat=c_lat, hidden=hidden, n_attn=0)

    def forward(self, x):
        return self.m(x)                                           # xh, bits/sym, y_q


def train_baseline(data, dev, lam_rate, proj, epochs=20, batch=64):
    m = BaselineCodec().to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    factor = (C.TINY["c_lat"] * 32) / (C.N_CH * C.N_T)             # bits/sym -> bpp
    for _ in range(epochs):
        for x, y, tgt in C.iterate(data["eeg_train"], data["lab_train"], data["tgt_train"],
                                   batch=batch, shuffle=True, dev=dev):
            xh, bps, _ = m(x)
            loss = F.mse_loss(xh, x) + lam_rate * (bps * factor) + LAM_TASK * _cos_task(proj, xh, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def train_stacked(data, dev, lambdas, proj, epochs=22, batch=64):
    m = StackedCodec(n_levels=len(lambdas)).to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(epochs):
        for x, y, tgt in C.iterate(data["eeg_train"], data["lab_train"], data["tgt_train"],
                                   batch=batch, shuffle=True, dev=dev):
            level = int(torch.randint(0, len(lambdas), (1,)))
            xh, by, bz = m(x, level)
            bpp = m.total_bpp(by, bz, x.shape[0])
            loss = F.mse_loss(xh, x) + lambdas[level] * bpp + LAM_TASK * _cos_task(proj, xh, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_baseline(m, data, dev, proj, n_disk=48):
    m.eval()
    factor = (C.TINY["c_lat"] * 32) / (C.N_CH * C.N_T)
    embs, mse, bps, n = [], 0.0, 0.0, 0
    lat = []
    for (x,) in C.iterate(data["eeg_test"], batch=128, shuffle=False, dev=dev):
        y = m.m.encoder(x); yq = torch.round(y); xh = m.m.decoder(yq)
        embs.append(proj(xh).cpu()); lat.append(yq.cpu())
        mse += F.mse_loss(xh, x, reduction="sum").item()
        bps += m.m.prior.bits(yq).item() * x.shape[0]; n += x.shape[0]
    ret = C.retrieval_topk(torch.cat(embs), data["prototypes"], data["lab_test"])
    bpp = bps / n * factor
    lat = torch.cat(lat)[:n_disk].numpy().astype(np.int64)
    # shipped on-disk: savez_compressed of int16 latents
    buf = io.BytesIO(); np.savez_compressed(buf, latents=lat.astype(np.int16))
    savez_bpp = buf.getbuffer().nbytes * 8 / (n_disk * C.N_CH * C.N_T)
    # A1 applied to baseline prior (isolates serialization gain)
    rc = range_code_latents(lat, m.m.prior.log_b.detach().cpu().numpy())
    rc_bpp = rc["actual_bits"] / (n_disk * C.N_CH * C.N_T)
    return {"bpp_modeled": round(bpp, 4), "mse": round(mse / (n * C.N_CH * C.N_T), 4),
            "top1": ret["top1"], "top5": ret["top5"],
            "bpp_savez": round(savez_bpp, 4), "ratio_savez": round(16 / savez_bpp, 1),
            "bpp_rangecoded": round(rc_bpp, 4), "ratio_rangecoded": round(16 / rc_bpp, 1),
            "rc_roundtrip": rc["roundtrip_ok"]}


@torch.no_grad()
def eval_stacked(m, data, dev, proj, level, n_disk=48):
    m.eval()
    g = m.gain(level)
    embs, mse, by_sum, bz_sum, n = [], 0.0, 0.0, 0.0, 0
    yqs, zqs = [], []
    for (x,) in C.iterate(data["eeg_test"], batch=128, shuffle=False, dev=dev):
        y = m.enc(x) * g; yq = torch.round(y)
        z = m.h_enc(y); zq = torch.round(z)
        w, mu, b = m._params(zq, y.shape[-1])
        by = m._mix_bits(yq, w, mu, b); bz = m.zprior.bits(zq)
        xh = m.dec(yq / g)
        embs.append(proj(xh).cpu()); yqs.append(yq.cpu()); zqs.append(zq.cpu())
        mse += F.mse_loss(xh, x, reduction="sum").item()
        by_sum += by.sum().item(); bz_sum += bz.sum().item(); n += x.shape[0]
    ret = C.retrieval_topk(torch.cat(embs), data["prototypes"], data["lab_test"])
    bpp = (by_sum + bz_sum) / (n * C.N_CH * C.N_T)

    # ---- real on-disk (A1) with the mixture-conditional pmf for y + factorized for z ----
    yq = torch.cat(yqs)[:n_disk]; zq = torch.cat(zqs)[:n_disk]
    w, mu, b = m._params(zq.to(dev), yq.shape[-1])
    w, mu, b = w.cpu().numpy(), mu.cpu().numpy(), b.cpu().numpy()   # (n,K,c_lat,T)
    yq_np = yq.numpy().astype(np.int64)
    yv = np.arange(int(yq_np.min()) - 1, int(yq_np.max()) + 2)
    # pmf per y-symbol over yv: sum_k w * (CDF(v+.5)-CDF(v-.5))
    def lap_cdf(x, mm, bb):
        z = x - mm
        return 0.5 + 0.5 * np.sign(z) * (1 - np.exp(-np.abs(z) / bb))
    nD, K, Cc, T = w.shape
    # flatten symbols in (n, c_lat, T) order
    w_f = np.transpose(w, (0, 2, 3, 1)).reshape(-1, K)             # (M,K)
    mu_f = np.transpose(mu, (0, 2, 3, 1)).reshape(-1, K)
    b_f = np.transpose(b, (0, 2, 3, 1)).reshape(-1, K)
    y_f = np.transpose(yq_np, (0, 1, 2)).reshape(-1)               # (M,)
    V = yv[None, None, :]
    pmf = (w_f[:, :, None] * (lap_cdf(V + 0.5, mu_f[:, :, None], b_f[:, :, None])
                              - lap_cdf(V - 0.5, mu_f[:, :, None], b_f[:, :, None]))).sum(1)
    pmf = np.clip(pmf, 1e-12, None)
    ry = code_and_measure(y_f, pmf, yv)
    # z symbols with factorized prior
    zq_np = zq.numpy().astype(np.int64)
    zv = np.arange(int(zq_np.min()) - 1, int(zq_np.max()) + 2)
    zpmf_by_ch = m.zprior.pmf_np(zv)                               # (hyper_ch, V)
    hc = zq_np.shape[1]
    z_f = zq_np.reshape(-1)
    ch_idx = np.tile(np.repeat(np.arange(hc), zq_np.shape[2]), zq_np.shape[0])
    zpmf = zpmf_by_ch[ch_idx]
    rz = code_and_measure(z_f, zpmf, zv)
    disk_bits = ry["actual_bits"] + rz["actual_bits"]
    rc_bpp = disk_bits / (n_disk * C.N_CH * C.N_T)                 # bits per EEG sample
    return {"level": level, "bpp_modeled": round(bpp, 4), "mse": round(mse / (n * C.N_CH * C.N_T), 4),
            "top1": ret["top1"], "top5": ret["top5"],
            "bpp_rangecoded": round(rc_bpp, 4), "ratio_rangecoded": round(16 / rc_bpp, 1),
            "rc_roundtrip": bool(ry["roundtrip_ok"] and rz["roundtrip_ok"]),
            "hyper_side_bits_frac": round(rz["actual_bits"] / disk_bits, 3)}


def _interp(xs, ys, x):
    pts = sorted(zip(xs, ys)); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
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
    C.banner(f"STACKED codec vs current NeuroZip (origin/main)  (device={dev})")
    data = C.make_dataset(n_concepts=80, trials_per_concept=30, noise=0.3,
                          task_scale=0.7, distractor_scale=1.2, seed=0)
    proj = train_projector(data, dev)

    base_lams = (0.02, 0.06, 0.15, 0.4)
    baseline = []
    for lam in base_lams:
        m = train_baseline(data, dev, lam, proj)
        baseline.append({"lambda": lam, **eval_baseline(m, data, dev, proj)})

    stack_lams = (0.02, 0.06, 0.15, 0.4)
    sm = train_stacked(data, dev, stack_lams, proj)                # ONE model
    stacked = [eval_stacked(sm, data, dev, proj, lvl) for lvl in range(len(stack_lams))]

    # matched-distortion head-to-head: at each baseline MSE, read stacked's real
    # on-disk ratio + retrieval by interpolating the stacked curve on MSE.
    s_mse = [s["mse"] for s in stacked]
    s_rc = [s["ratio_rangecoded"] for s in stacked]
    s_top5 = [s["top5"] for s in stacked]
    matched = []
    for b in baseline:
        if min(s_mse) <= b["mse"] <= max(s_mse):
            matched.append({
                "mse": b["mse"],
                "baseline_ratio_savez_shipped": b["ratio_savez"],
                "baseline_ratio_rangecoded": b["ratio_rangecoded"],
                "stacked_ratio_rangecoded": round(_interp(s_mse, s_rc, b["mse"]), 1),
                "on_disk_gain_vs_shipped_x": round(_interp(s_mse, s_rc, b["mse"]) / b["ratio_savez"], 2),
                "baseline_top5": b["top5"],
                "stacked_top5": round(_interp(s_mse, s_top5, b["mse"]), 4),
            })

    results = {
        "baseline_origin_main": {"n_models": len(baseline), "points": baseline},
        "stacked": {"n_models": 1, "points": stacked},
        "matched_distortion": matched,
        "stack_contents": ["A1 range-coding", "A2+A3 mixture hyperprior",
                           "A5 variable-rate (1 model)", "A6 cosine task"],
        "excluded": ["A4 VQ (exclusive with scalar+hyperprior; extrapolated-only win)"],
    }
    print("RESULTS_JSON:" + json.dumps(results))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
