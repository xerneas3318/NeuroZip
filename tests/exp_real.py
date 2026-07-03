"""Full benchmark on REAL THINGS-EEG (Haitao999/things-eeg, sub-01).

Runs the A-series + the stacked-vs-origin/main head-to-head on the actual
dataset and dumps ALL stats. Uses the repo's real loader (`data.py`,
normalization + precomputed LAION-2B CLIP features), the byte-identical
`codec.EEGCodec`, and the recommended attention projector as the frozen judge.

Config = origin/main v4: codec conv-only c_lat=32 / hidden=128; projector
attention hidden=128 / n_attn=2. Codecs train on single-trial train epochs;
evaluation is trial-averaged over the 80 test repetitions (the paper protocol),
with retrieval against both the 200-concept CLIP IMAGE gallery and the CLIP
TEXT gallery.

Run:  .venv/bin/python tests/exp_real.py
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as C

# ---- reconfigure the shared harness to origin/main's real codec size ----
C.TINY = dict(hidden=128, c_lat=32, n_attn=0)
C_LAT, HIDDEN = 32, 128

import data as td                                     # repo real loader
from clip_proj import EEGProjector, info_nce
from codec import EEGCodec, quantize
from exp_priors import (Hyperprior, MixturePrior, fit_rate_model,
                        eval_bits_per_symbol, range_code_latents,
                        train_fidelity_codec)
from exp_vq import train_scalar, eval_scalar, train_vq, eval_vq
from exp_variable_rate import (train_variable, eval_variable, train_fixed,
                               eval_fixed, LAMBDAS as VR_LAMBDAS)
from exp_stacked import (StackedCodec, BaselineCodec, train_baseline,
                         train_stacked, code_and_measure)

TRAIN_CAP = 16000              # codec-training subsample (kept light)
PROJ_CAP = None                # judge trains on the FULL train set (like train.py)
PROJ_EPOCHS = 8
N_DISK = 64                    # test epochs used for the real on-disk measurement
BATCH = 128                    # small — GPU is shared/starved
TAU = 0.07


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_real():
    tr = td.ThingsEEG(split="train", max_epochs=PROJ_CAP)
    te = td.ThingsEEG(split="test")
    te_avg, te_text, te_img = te.trial_averaged()      # (200,63,250)
    img_gallery = te.concept_clip_img().float()        # (200,512)
    text_gallery, concepts = te.concept_clip_text()    # (200,512), list
    # map each averaged-test concept row -> its index in the text gallery
    cidx = {c: i for i, c in enumerate(concepts)}
    lab_test = torch.tensor([cidx[str(t)] for t in te_text]).long()
    data = {
        "eeg_train": torch.from_numpy(tr.eeg).float(),
        "tgt_train": torch.from_numpy(tr.clip_img).float(),
        "lab_train": torch.zeros(len(tr.eeg)).long(),   # unused (batchwise InfoNCE)
        "eeg_test": te_avg.float(),
        "lab_test": lab_test,
        "prototypes": img_gallery,                      # image gallery (retrieval)
        "tgt_test": img_gallery[lab_test],
        "text_gallery": text_gallery.float(),
        "n_concepts": img_gallery.shape[0],
        "meta": {"train_epochs_used": int(len(tr.eeg)), "train_total": tr.n_concepts * tr.n_reps,
                 "test_concepts": te.n_concepts, "test_reps": te.n_reps},
    }
    return data


# --------------------------------------------------------------------------- #
# Judge (attention projector) + retrieval
# --------------------------------------------------------------------------- #

def train_projector(data, dev, epochs=PROJ_EPOCHS, batch=BATCH):
    """Mirror train.py: full train set, AdamW + cosine schedule, InfoNCE."""
    proj = EEGProjector(hidden=HIDDEN, n_attn=2).to(dev).train()
    opt = torch.optim.AdamW(proj.parameters(), lr=3e-4, weight_decay=1e-4)
    n = data["eeg_train"].shape[0]
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for ep in range(epochs):
        for x, tgt in C.iterate(data["eeg_train"], data["tgt_train"], batch=batch,
                                shuffle=True, dev=dev):
            loss = info_nce(proj(x), tgt, temperature=TAU)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc = retr_both(project(proj, data["eeg_test"], dev), data)
        print(f"[proj] epoch {ep+1}/{epochs} img_top1={acc['image']['top1']:.3f} "
              f"img_top5={acc['image']['top5']:.3f}", flush=True)
    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


@torch.no_grad()
def project(proj, eeg, dev, batch=256):
    out = []
    for (x,) in C.iterate(eeg, batch=batch, shuffle=False, dev=dev):
        out.append(proj(x).cpu())
    return torch.cat(out)


def retr_both(emb, data):
    img = C.retrieval_topk(emb, data["prototypes"], data["lab_test"])
    txt = C.retrieval_topk(emb, data["text_gallery"], data["lab_test"])
    return {"image": img, "text": txt}


# --------------------------------------------------------------------------- #
# Real on-disk helpers (A1)
# --------------------------------------------------------------------------- #

def savez_bpp(lat_int_np, n_ep):
    buf = io.BytesIO(); np.savez_compressed(buf, latents=lat_int_np.astype(np.int16))
    return buf.getbuffer().nbytes * 8 / (n_ep * C.N_CH * C.N_T)


# --------------------------------------------------------------------------- #
# Comprehensive codec eval (baseline factorized OR stacked mixture-hyperprior)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def eval_baseline_full(m, data, dev, proj):
    m.eval()
    factor = (C_LAT * 32) / (C.N_CH * C.N_T)
    recon_emb, mse, bps, n, lats = [], 0.0, 0.0, 0, []
    for (x,) in C.iterate(data["eeg_test"], batch=128, shuffle=False, dev=dev):
        y = m.m.encoder(x); yq = torch.round(y); xh = m.m.decoder(yq)
        recon_emb.append(proj(xh).cpu()); lats.append(yq.cpu())
        mse += F.mse_loss(xh, x, reduction="sum").item()
        bps += m.m.prior.bits(yq).item() * x.shape[0]; n += x.shape[0]
    ret = retr_both(torch.cat(recon_emb), data)
    lat = torch.cat(lats)[:N_DISK].numpy().astype(np.int64)
    sz = savez_bpp(lat, N_DISK)
    rc = range_code_latents(lat, m.m.prior.log_b.detach().cpu().numpy())
    rc_bpp = rc["actual_bits"] / (N_DISK * C.N_CH * C.N_T)
    return {"mse": round(mse / (n * C.N_CH * C.N_T), 5),
            "bpp_modeled": round(bps / n * factor, 4),
            "retrieval": ret,
            "bpp_savez": round(sz, 4), "ratio_savez": round(16 / sz, 1),
            "bpp_rangecoded": round(rc_bpp, 4), "ratio_rangecoded": round(16 / rc_bpp, 1),
            "rc_roundtrip": rc["roundtrip_ok"]}


@torch.no_grad()
def eval_stacked_full(m, data, dev, proj, level):
    m.eval(); g = m.gain(level)
    recon_emb, mse, by_sum, bz_sum, n = [], 0.0, 0.0, 0.0, 0
    yqs, zqs = [], []
    for (x,) in C.iterate(data["eeg_test"], batch=128, shuffle=False, dev=dev):
        y = m.enc(x) * g; yq = torch.round(y)
        z = m.h_enc(y); zq = torch.round(z)
        w, mu, b = m._params(zq, y.shape[-1])
        by = m._mix_bits(yq, w, mu, b); bz = m.zprior.bits(zq)
        xh = m.dec(yq / g)
        recon_emb.append(proj(xh).cpu()); yqs.append(yq.cpu()); zqs.append(zq.cpu())
        mse += F.mse_loss(xh, x, reduction="sum").item()
        by_sum += by.sum().item(); bz_sum += bz.sum().item(); n += x.shape[0]
    ret = retr_both(torch.cat(recon_emb), data)
    bpp = (by_sum + bz_sum) / (n * C.N_CH * C.N_T)

    yq = torch.cat(yqs)[:N_DISK]; zq = torch.cat(zqs)[:N_DISK]
    w, mu, b = m._params(zq.to(dev), yq.shape[-1])
    w, mu, b = w.cpu().numpy(), mu.cpu().numpy(), b.cpu().numpy()
    yq_np = yq.numpy().astype(np.int64)
    yv = np.arange(int(yq_np.min()) - 1, int(yq_np.max()) + 2)

    def lap(x, mm, bb):
        z = x - mm
        return 0.5 + 0.5 * np.sign(z) * (1 - np.exp(-np.abs(z) / bb))
    K = w.shape[1]
    w_f = np.transpose(w, (0, 2, 3, 1)).reshape(-1, K)
    mu_f = np.transpose(mu, (0, 2, 3, 1)).reshape(-1, K)
    b_f = np.transpose(b, (0, 2, 3, 1)).reshape(-1, K)
    y_f = yq_np.reshape(-1)
    V = yv[None, None, :]
    pmf = (w_f[:, :, None] * (lap(V + 0.5, mu_f[:, :, None], b_f[:, :, None])
                              - lap(V - 0.5, mu_f[:, :, None], b_f[:, :, None]))).sum(1)
    ry = code_and_measure(y_f, np.clip(pmf, 1e-12, None), yv)
    zq_np = zq.numpy().astype(np.int64)
    zv = np.arange(int(zq_np.min()) - 1, int(zq_np.max()) + 2)
    zpmf_ch = m.zprior.pmf_np(zv)
    hc = zq_np.shape[1]
    ch_idx = np.tile(np.repeat(np.arange(hc), zq_np.shape[2]), zq_np.shape[0])
    rz = code_and_measure(zq_np.reshape(-1), zpmf_ch[ch_idx], zv)
    disk_bits = ry["actual_bits"] + rz["actual_bits"]
    rc_bpp = disk_bits / (N_DISK * C.N_CH * C.N_T)
    return {"level": level, "mse": round(mse / (n * C.N_CH * C.N_T), 5),
            "bpp_modeled": round(bpp, 4), "retrieval": ret,
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


# --------------------------------------------------------------------------- #
# Main: run every stage, accumulate all stats
# --------------------------------------------------------------------------- #

def stage(results, name, fn):
    t = time.time()
    try:
        results[name] = fn()
        results[name + "_secs"] = round(time.time() - t, 1)
        print(f"[ok] {name} ({results[name+'_secs']}s)", flush=True)
    except Exception as e:
        import traceback
        results[name] = {"ERROR": f"{type(e).__name__}: {e}"}
        print(f"[FAIL] {name}: {e}", flush=True); traceback.print_exc()
    # incremental dump so partial runs still yield stats
    Path("tests/results/REAL_all.json").write_text(json.dumps(results, indent=1))


def main():
    C.seed_all(0)
    dev = C.device()
    C.banner(f"REAL THINGS-EEG benchmark (sub-01)  device={dev}  codec c_lat={C_LAT}/hidden={HIDDEN}")
    results = {"config": {"c_lat": C_LAT, "hidden": HIDDEN, "projector": "attn n_attn=2 hidden=128",
                          "train_cap": TRAIN_CAP, "n_disk_epochs": N_DISK,
                          "eval": "trial-averaged over 80 reps"}}

    data = load_real()
    results["data"] = data["meta"]
    print(f"[data] {data['meta']}", flush=True)

    proj = train_projector(data, dev)
    ceil = retr_both(project(proj, data["eeg_test"], dev), data)
    results["projector_ceiling_no_compression"] = ceil
    results["chance_top1"] = round(1 / data["n_concepts"], 4)
    print(f"[ceiling] {ceil}", flush=True)

    # Judge is trained on the FULL set; codec stages use a light random subsample.
    full_n = data["eeg_train"].shape[0]
    if full_n > TRAIN_CAP:
        perm = torch.randperm(full_n)[:TRAIN_CAP]
        data["eeg_train"] = data["eeg_train"][perm].contiguous()
        data["tgt_train"] = data["tgt_train"][perm].contiguous()
        data["lab_train"] = data["lab_train"][perm].contiguous()
        import gc; gc.collect()
    results["data"]["codec_train_epochs"] = int(data["eeg_train"].shape[0])

    # ---------- stacked vs origin/main (PRIMARY) ----------
    def run_stacked_vs_main():
        base_lams = (0.02, 0.06, 0.15, 0.4)
        baseline = []
        for lam in base_lams:
            m = train_baseline(data, dev, lam, proj, epochs=10)
            baseline.append({"lambda": lam, **eval_baseline_full(m, data, dev, proj)})
        stack_lams = (0.02, 0.06, 0.15, 0.4)
        sm = train_stacked(data, dev, stack_lams, proj, epochs=12)
        stacked = [eval_stacked_full(sm, data, dev, proj, lvl) for lvl in range(len(stack_lams))]
        # matched-distortion: baseline savez vs stacked range-coded at same MSE
        s_mse = [s["mse"] for s in stacked]; s_rc = [s["ratio_rangecoded"] for s in stacked]
        s_it5 = [s["retrieval"]["image"]["top5"] for s in stacked]
        matched = []
        for b in baseline:
            if min(s_mse) <= b["mse"] <= max(s_mse):
                matched.append({"mse": b["mse"],
                                "baseline_ratio_savez_shipped": b["ratio_savez"],
                                "baseline_ratio_rangecoded": b["ratio_rangecoded"],
                                "stacked_ratio_rangecoded": round(_interp(s_mse, s_rc, b["mse"]), 1),
                                "on_disk_gain_vs_shipped_x": round(_interp(s_mse, s_rc, b["mse"]) / b["ratio_savez"], 2),
                                "baseline_img_top5": b["retrieval"]["image"]["top5"],
                                "stacked_img_top5": round(_interp(s_mse, s_it5, b["mse"]), 4)})
        return {"baseline_origin_main": {"n_models": len(baseline), "points": baseline},
                "stacked": {"n_models": 1, "points": stacked},
                "matched_distortion": matched}
    stage(results, "STACKED_vs_origin_main", run_stacked_vs_main)

    # ---------- A1/A2/A3 priors ----------
    def run_priors():
        codec = C.make_codec(**C.TINY)
        train_fidelity_codec(codec, data, dev, epochs=10, lam_rate=0.05)
        lat = C.collect_latents(codec, data["eeg_test"], dev)
        c_lat = lat.shape[1]
        with torch.no_grad():
            fact = float(codec.prior.bits(lat.to(dev)).item())
        rc = range_code_latents(lat[:N_DISK].numpy().astype(np.int64),
                                codec.prior.log_b.detach().cpu().numpy())
        actual = rc["actual_bits"] / rc["n_symbols"]
        sz = savez_bpp(lat[:N_DISK].numpy().astype(np.int64), N_DISK)
        lat_train = C.collect_latents(codec, data["eeg_train"], dev)
        hyper = fit_rate_model(Hyperprior(c_lat=c_lat), lat_train, dev, epochs=30)
        mix = fit_rate_model(MixturePrior(c_lat=c_lat, k=4), lat_train, dev, epochs=50)
        return {"A1_real_on_disk": {
                    "modeled_bits_per_symbol": round(fact, 4),
                    "range_coded_bits_per_symbol": round(actual, 4),
                    "coder_overhead_pct": round(100 * (actual / (rc["theoretical_bits"] / rc["n_symbols"]) - 1), 3),
                    "roundtrip_ok": rc["roundtrip_ok"],
                    "bpp_modeled": round(C.latent_bpp(fact, c_lat), 4),
                    "bpp_rangecoded": round(C.latent_bpp(actual, c_lat), 4),
                    "bpp_savez": round(sz, 4),
                    "ratio_rangecoded": round(16 / C.latent_bpp(actual, c_lat), 1),
                    "ratio_savez": round(16 / sz, 1)},
                "A2_hyperprior": {"factorized_bps": round(fact, 4),
                                  "hyperprior_bps": round(eval_bits_per_symbol(hyper, lat, dev), 4),
                                  "rate_reduction_pct": round(100 * (1 - eval_bits_per_symbol(hyper, lat, dev) / fact), 2)},
                "A3_mixture": {"single_laplace_bps": round(fact, 4),
                               "mixture_bps": round(eval_bits_per_symbol(mix, lat, dev), 4),
                               "rate_reduction_pct": round(100 * (1 - eval_bits_per_symbol(mix, lat, dev) / fact), 2)}}
    stage(results, "A1_A2_A3_priors", run_priors)

    # ---------- A4 VQ ----------
    def run_vq():
        scal, vq = [], []
        for lam in (0.02, 0.08, 0.3, 1.0):
            m = train_scalar(data, dev, lam, epochs=10)
            mse, bpp = eval_scalar(m, data, dev)
            scal.append({"lambda": lam, "bpp": round(bpp, 4), "mse": round(mse, 4), "ratio": round(16 / bpp, 1)})
        for k in (64, 256, 1024):
            m = train_vq(data, dev, k, epochs=10)
            mse, bpp, ent = eval_vq(m, data, dev)
            vq.append({"K": k, "code_entropy_bits": round(ent, 3), "bpp": round(bpp, 4),
                       "mse": round(mse, 4), "ratio": round(16 / bpp, 1)})
        return {"scalar_rd": scal, "vq_rd": vq}
    stage(results, "A4_vq", run_vq)

    # ---------- A5 variable-rate ----------
    def run_varrate():
        vr = train_variable(data, dev, epochs=12)
        var_pts = []
        for r in range(len(VR_LAMBDAS)):
            mse, bpp = eval_variable(vr, data, dev, r)
            var_pts.append({"level": r, "lambda": VR_LAMBDAS[r], "bpp": round(bpp, 4),
                            "mse": round(mse, 4), "ratio": round(16 / bpp, 1)})
        fixed_pts = []
        for lam in VR_LAMBDAS:
            m = train_fixed(data, dev, lam, epochs=12)
            mse, bpp = eval_fixed(m, data, dev)
            fixed_pts.append({"lambda": lam, "bpp": round(bpp, 4), "mse": round(mse, 4), "ratio": round(16 / bpp, 1)})
        return {"variable_rate_one_model": var_pts, "fixed_rate_dedicated": fixed_pts}
    stage(results, "A5_variable_rate", run_varrate)

    # ---------- A6 contrastive vs cosine vs fidelity ----------
    def run_contrastive():
        out = {}
        for mode in ("fidelity", "cosine", "contrastive"):
            m = _train_task_real(mode, data, dev, proj, lam_rate=0.06, epochs=10)
            out[mode] = _eval_task_real(m, data, dev, proj)
        return out
    stage(results, "A6_contrastive_vs_cosine", run_contrastive)

    results["stack_contents"] = ["A1 range-coding", "A2+A3 mixture hyperprior",
                                 "A5 variable-rate (1 model)", "A6 cosine task"]
    print("RESULTS_JSON:" + json.dumps(results))
    print(json.dumps(results, indent=2))
    return results


# --- A6 helpers on real data (cosine-to-target / batchwise InfoNCE) ---

class _Task(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = EEGCodec(c_lat=C_LAT, hidden=HIDDEN, n_attn=0)

    def forward(self, x):
        y = self.m.encoder(x); yq = quantize(y, self.training)
        return self.m.decoder(yq), self.m.prior.bits(yq)


def _train_task_real(mode, data, dev, proj, lam_rate, epochs=10, batch=128, lam_task=3.0):
    m = _Task().to(dev).train()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    factor = (C_LAT * 32) / (C.N_CH * C.N_T)
    for _ in range(epochs):
        for x, tgt in C.iterate(data["eeg_train"], data["tgt_train"], batch=batch, shuffle=True, dev=dev):
            xh, bps = m(x)
            loss = F.mse_loss(xh, x) + lam_rate * (bps * factor)
            if mode == "cosine":
                loss = loss + lam_task * (1 - F.cosine_similarity(proj(xh), F.normalize(tgt, dim=-1), dim=-1)).mean()
            elif mode == "contrastive":
                loss = loss + lam_task * info_nce(proj(xh), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def _eval_task_real(m, data, dev, proj):
    m.eval()
    factor = (C_LAT * 32) / (C.N_CH * C.N_T)
    emb, mse, bps, n = [], 0.0, 0.0, 0
    for (x,) in C.iterate(data["eeg_test"], batch=128, shuffle=False, dev=dev):
        y = m.m.encoder(x); yq = torch.round(y); xh = m.m.decoder(yq)
        emb.append(proj(xh).cpu()); mse += F.mse_loss(xh, x, reduction="sum").item()
        bps += m.m.prior.bits(yq).item() * x.shape[0]; n += x.shape[0]
    ret = retr_both(torch.cat(emb), data)
    return {"bpp_modeled": round(bps / n * factor, 4), "mse": round(mse / (n * C.N_CH * C.N_T), 5),
            "retrieval": ret}


if __name__ == "__main__":
    main()
