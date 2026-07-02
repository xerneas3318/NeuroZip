"""Group A / entropy-model family: A1 real on-disk ratio, A2 hyperprior,
A3 learned mixture prior.

All three are answered on ONE set of codec latents so the comparison is
controlled (only the entropy model changes). Pipeline:

  1. Train a tiny fidelity codec (rate + recon) a few epochs on the surrogate.
  2. Collect its integer test latents.
  3. A1  -- range-code those latents with the codec's OWN factorized Laplace
           prior. Measure the actual byte count and compare to (i) the modeled
           bpp NeuroZip reports, (ii) np.savez_compressed (what save_nz() ships
           today), (iii) raw int16 and raw fp16. Round-trip is verified.
  4. A2  -- fit a scale-hyperprior entropy model on the SAME latents; report
           bits/symbol (incl. the side-info z bits) vs the factorized prior.
  5. A3  -- fit a per-channel mixture-of-Laplaces prior on the same latents;
           report bits/symbol vs the single-Laplace factorized prior.

Run:  .venv/bin/python tests/exp_priors.py
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


# --------------------------------------------------------------------------- #
# A tiny arithmetic (range) coder -- textbook 32-bit WNC coder, integer freqs.
# Python big-ints mean no register overflow; we only mask to 32 bits on shift.
# --------------------------------------------------------------------------- #

_TOP = 1 << 32
_HALF = 1 << 31
_QTR = 1 << 30
_3QTR = 3 << 30
_MASK = _TOP - 1


class _BitOut:
    def __init__(self):
        self.bits = bytearray()

    def put(self, b):
        self.bits.append(b & 1)

    def to_bytes(self):
        pad = (-len(self.bits)) % 8
        bits = list(self.bits) + [0] * pad
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for k in range(8):
                byte = (byte << 1) | bits[i + k]
            out.append(byte)
        return bytes(out)


class _BitIn:
    def __init__(self, data, nbits):
        self.bits = []
        for byte in data:
            for k in range(7, -1, -1):
                self.bits.append((byte >> k) & 1)
        self.n = nbits
        self.i = 0

    def get(self):
        b = self.bits[self.i] if self.i < len(self.bits) else 0
        self.i += 1
        return b


def _encode(cum, freq, total):
    """cum/freq/total: per-symbol (cumulative-low, frequency, table-total)."""
    out = _BitOut()
    low, high, pending = 0, _MASK, 0

    def emit(bit):
        out.put(bit)
        for _ in range(pending[0]):
            out.put(bit ^ 1)
        pending[0] = 0

    pending = [0]
    for cl, fr, tot in zip(cum, freq, total):
        rng = high - low + 1
        high = low + (rng * (cl + fr)) // tot - 1
        low = low + (rng * cl) // tot
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QTR and high < _3QTR:
                pending[0] += 1
                low -= _QTR
                high -= _QTR
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) & _MASK) | 1
    pending[0] += 1
    emit(0 if low < _QTR else 1)
    return out.to_bytes(), len(out.bits)


def _decode(data, nbits, n, tables):
    """tables: per-symbol callable returning the model at step i given decoded.
    Here the model is position-dependent but data-independent, so `tables` is a
    list of (values, cumfreq, total) per position."""
    bi = _BitIn(data, nbits)
    code = 0
    for _ in range(32):
        code = ((code << 1) | bi.get()) & _MASK
    low, high = 0, _MASK
    out = []
    for i in range(n):
        values, cumfreq, total = tables[i]
        rng = high - low + 1
        # find symbol whose cumulative interval contains value
        val = (((code - low) + 1) * total - 1) // rng
        # binary search in cumfreq (monotone), cumfreq has len(values)+1 entries
        lo, hi = 0, len(values) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cumfreq[mid] <= val:
                lo = mid
            else:
                hi = mid - 1
        s = lo
        cl, ch = cumfreq[s], cumfreq[s + 1]
        high = low + (rng * ch) // total - 1
        low = low + (rng * cl) // total
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                code -= _HALF
            elif low >= _QTR and high < _3QTR:
                low -= _QTR
                high -= _QTR
                code -= _QTR
            else:
                break
            low = (low << 1) & _MASK
            high = ((high << 1) & _MASK) | 1
            code = ((code << 1) & _MASK) | bi.get()
        out.append(values[s])
    return out


def _laplace_cdf(z, b):
    return 0.5 + 0.5 * np.sign(z) * (1.0 - np.exp(-np.abs(z) / b))


def range_code_latents(latents_np, log_b, precision_bits=16):
    """Range-code integer latents (N,C,T) with a per-channel Laplace model.

    Returns dict with actual_bits, theoretical_bits, roundtrip_ok.
    """
    N, Cc, T = latents_np.shape
    b = np.log1p(np.exp(log_b))  # softplus, (C,)
    b = np.clip(b, 1e-4, None)
    total = 1 << precision_bits
    vmin = int(latents_np.min()) - 1
    vmax = int(latents_np.max()) + 1
    values = np.arange(vmin, vmax + 1)

    # Per-channel integer frequency tables + cumulative.
    chan_tables = []
    for c in range(Cc):
        p = _laplace_cdf(values + 0.5, b[c]) - _laplace_cdf(values - 0.5, b[c])
        p = np.clip(p, 1e-12, None)
        p = p / p.sum()
        freq = np.maximum((p * total).astype(np.int64), 1)
        # fix rounding so freqs sum exactly to total
        diff = total - int(freq.sum())
        freq[int(np.argmax(freq))] += diff
        assert freq.min() >= 1 and freq.sum() == total
        cumfreq = np.concatenate([[0], np.cumsum(freq)])
        chan_tables.append((values, freq, cumfreq))

    # Flatten symbols in (N, C, T) order; each position uses its channel table.
    sym_cum, sym_freq, sym_tot, dec_tables, gold = [], [], [], [], []
    theo_bits = 0.0
    for n in range(N):
        for c in range(Cc):
            values_c, freq_c, cum_c = chan_tables[c]
            row = latents_np[n, c]
            for v in row:
                idx = int(v - vmin)
                sym_cum.append(int(cum_c[idx]))
                sym_freq.append(int(freq_c[idx]))
                sym_tot.append(total)
                dec_tables.append((values_c, cum_c, total))
                gold.append(int(v))
                theo_bits += -math.log2(freq_c[idx] / total)

    data, nbits = _encode(sym_cum, sym_freq, sym_tot)
    decoded = _decode(data, nbits, len(gold), dec_tables)
    roundtrip = (decoded == gold)
    return {
        "actual_bits": len(data) * 8,
        "coder_stream_bits": nbits,
        "theoretical_bits": theo_bits,
        "n_symbols": len(gold),
        "roundtrip_ok": bool(roundtrip),
    }


# --------------------------------------------------------------------------- #
# A2: scale hyperprior on fixed latents
# --------------------------------------------------------------------------- #

class _FactorizedScale(nn.Module):
    """Per-channel Laplace scale prior (for the hyper-latent z)."""
    def __init__(self, ch):
        super().__init__()
        self.log_b = nn.Parameter(torch.zeros(ch))

    def bits(self, z):
        b = F.softplus(self.log_b)[None, :, None].clamp_min(1e-4)
        cdf = lambda x: 0.5 + 0.5 * torch.sign(x) * (1 - torch.exp(-x.abs() / b))
        p = (cdf(z + 0.5) - cdf(z - 0.5)).clamp_min(1e-9)
        return -torch.log2(p)


class Hyperprior(nn.Module):
    """Predict a per-position Laplace scale for y from a small hyper-latent z."""
    def __init__(self, c_lat=16, hyper_ch=16):
        super().__init__()
        self.h_enc = nn.Sequential(
            nn.Conv1d(c_lat, hyper_ch, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(hyper_ch, hyper_ch, 3, stride=2, padding=1),
        )
        self.h_dec = nn.Sequential(
            nn.ConvTranspose1d(hyper_ch, hyper_ch, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(hyper_ch, c_lat, 4, stride=2, padding=1),
        )
        self.zprior = _FactorizedScale(hyper_ch)

    def forward(self, y, training):
        z = self.h_enc(y)
        z_q = z + (torch.rand_like(z) - 0.5) if training else torch.round(z)
        log_scale = self.h_dec(z_q)
        log_scale = log_scale[..., :y.shape[-1]]
        b = F.softplus(log_scale).clamp_min(1e-4)
        cdf = lambda x: 0.5 + 0.5 * torch.sign(x) * (1 - torch.exp(-x.abs() / b))
        y_q = y + (torch.rand_like(y) - 0.5) if training else torch.round(y)
        py = (cdf(y_q + 0.5) - cdf(y_q - 0.5)).clamp_min(1e-9)
        bits_y = -torch.log2(py)
        bits_z = self.zprior.bits(z_q)
        return bits_y.sum(), bits_z.sum()


# --------------------------------------------------------------------------- #
# A3: per-channel mixture-of-Laplaces prior
# --------------------------------------------------------------------------- #

class MixturePrior(nn.Module):
    def __init__(self, c_lat=16, k=4):
        super().__init__()
        self.logit_w = nn.Parameter(torch.zeros(c_lat, k))
        self.log_b = nn.Parameter(torch.randn(c_lat, k) * 0.1)
        self.mu = nn.Parameter(torch.randn(c_lat, k) * 0.1)
        self.k = k

    def bits(self, y):
        # y: (N, C, T) integer/noisy
        w = torch.softmax(self.logit_w, dim=-1)              # (C,K)
        b = F.softplus(self.log_b).clamp_min(1e-3)           # (C,K)
        mu = self.mu                                         # (C,K)
        yv = y.unsqueeze(-1)                                 # (N,C,T,1)
        w = w[None, :, None, :]
        b = b[None, :, None, :]
        mu = mu[None, :, None, :]

        def cdf(x):
            zz = (x - mu)
            return 0.5 + 0.5 * torch.sign(zz) * (1 - torch.exp(-zz.abs() / b))
        p = (cdf(yv + 0.5) - cdf(yv - 0.5)).clamp_min(1e-12)
        p = (w * p).sum(-1).clamp_min(1e-12)                 # (N,C,T)
        return -torch.log2(p)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def train_fidelity_codec(codec, data, dev, epochs=12, lam_rate=0.02, batch=64):
    codec.to(dev).train()
    opt = torch.optim.Adam(codec.parameters(), lr=2e-3)
    for ep in range(epochs):
        for (x,) in C.iterate(data["eeg_train"], batch=batch, shuffle=True, dev=dev):
            xh, bps, _ = codec(x)
            loss = F.mse_loss(xh, x) + lam_rate * bps
            opt.zero_grad(); loss.backward(); opt.step()
    return codec


def fit_rate_model(model, latents, dev, epochs, lr=5e-3, batch=128):
    model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = latents.shape[0]
    for ep in range(epochs):
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            y = latents[idx[i:i + batch]].to(dev)
            if isinstance(model, Hyperprior):
                by, bz = model(y, training=True)
                loss = (by + bz) / y.numel()
            else:
                loss = model.bits(y + (torch.rand_like(y) - 0.5)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_bits_per_symbol(model, latents, dev, batch=128):
    model.eval()
    tot_bits, tot_sym = 0.0, 0
    for i in range(0, latents.shape[0], batch):
        y = latents[i:i + batch].to(dev)
        if isinstance(model, Hyperprior):
            by, bz = model(y, training=False)
            tot_bits += float(by + bz)
        else:
            tot_bits += float(model.bits(torch.round(y)).sum())
        tot_sym += y.numel()
    return tot_bits / tot_sym


def main():
    C.seed_all(0)
    dev = C.device()
    C.banner(f"A1/A2/A3  entropy-model family  (device={dev})")
    data = C.make_dataset(n_concepts=40, trials_per_concept=40, noise=0.8, seed=0)

    codec = C.make_codec(**C.TINY)
    train_fidelity_codec(codec, data, dev, epochs=12, lam_rate=0.02)
    lat_test = C.collect_latents(codec, data["eeg_test"], dev)     # (Nte,16,32) int
    c_lat = lat_test.shape[1]
    n_ep = lat_test.shape[0]

    # ---- baseline factorized bits/symbol from the codec's own prior ----
    with torch.no_grad():
        fact_bps = float(codec.prior.bits(lat_test.to(dev)).item())
    fact_bpp = C.latent_bpp(fact_bps, c_lat)

    # ---- A1: real range coding with that same factorized prior ----
    n_code = min(128, n_ep)
    rc = range_code_latents(lat_test[:n_code].numpy().astype(np.int64),
                            codec.prior.log_b.detach().cpu().numpy())
    actual_bps = rc["actual_bits"] / rc["n_symbols"]
    actual_bpp = C.latent_bpp(actual_bps, c_lat)

    # on-disk baselines for the SAME n_code epochs
    sub = lat_test[:n_code].numpy().astype(np.int16)
    buf = io.BytesIO(); np.savez_compressed(buf, latents=sub)
    savez_bytes = buf.getbuffer().nbytes
    savez_bpp = C.latent_bpp(savez_bytes * 8 / (n_code * c_lat * 32), c_lat)
    raw_int16_bpp = C.latent_bpp(16.0, c_lat)          # 16 bits/symbol
    fp16_bpp = 16.0                                    # raw EEG fp16 reference

    # ---- A2: hyperprior on the same latents ----
    hyper = fit_rate_model(Hyperprior(c_lat=c_lat), data_lat := C.collect_latents(codec, data["eeg_train"], dev), dev, epochs=40)
    hyper_bps = eval_bits_per_symbol(hyper, lat_test, dev)
    hyper_bpp = C.latent_bpp(hyper_bps, c_lat)

    # ---- A3: mixture prior on the same latents ----
    mix = fit_rate_model(MixturePrior(c_lat=c_lat, k=4), data_lat, dev, epochs=60)
    mix_bps = eval_bits_per_symbol(mix, lat_test, dev)
    mix_bpp = C.latent_bpp(mix_bps, c_lat)

    results = {
        "n_epochs_coded": n_code,
        "latent_shape": [int(c_lat), 32],
        "A1_real_on_disk": {
            "modeled_bits_per_symbol": round(fact_bps, 4),
            "range_coded_bits_per_symbol": round(actual_bps, 4),
            "theoretical_bits_per_symbol_on_subset": round(rc["theoretical_bits"] / rc["n_symbols"], 4),
            "roundtrip_ok": rc["roundtrip_ok"],
            "coder_overhead_pct": round(100 * (actual_bps / (rc["theoretical_bits"] / rc["n_symbols"]) - 1), 3),
            "bpp_modeled": round(fact_bpp, 4),
            "bpp_range_coded_actual": round(actual_bpp, 4),
            "bpp_savez_compressed_today": round(savez_bpp, 4),
            "bpp_raw_int16": round(raw_int16_bpp, 4),
            "ratio_vs_fp16_range_coded": round(C.ratio_vs_fp16(actual_bpp), 1),
            "ratio_vs_fp16_savez_today": round(C.ratio_vs_fp16(savez_bpp), 1),
        },
        "A2_hyperprior": {
            "factorized_bits_per_symbol": round(fact_bps, 4),
            "hyperprior_bits_per_symbol": round(hyper_bps, 4),
            "rate_reduction_pct": round(100 * (1 - hyper_bps / fact_bps), 2),
            "bpp_factorized": round(fact_bpp, 4),
            "bpp_hyperprior": round(hyper_bpp, 4),
        },
        "A3_mixture_prior": {
            "single_laplace_bits_per_symbol": round(fact_bps, 4),
            "mixture_bits_per_symbol": round(mix_bps, 4),
            "rate_reduction_pct": round(100 * (1 - mix_bps / fact_bps), 2),
            "bpp_mixture": round(mix_bpp, 4),
        },
    }
    print("RESULTS_JSON:" + json.dumps(results))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
