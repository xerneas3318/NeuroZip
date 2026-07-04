"""Stacked codec vs the SHIPPED v4 checkpoints (real weights from the release).

Loads the actual shipped bundle (attention projector + fidelity/neurozip v4
low/med/high/xhigh codecs + norm stats + scores) and benchmarks the full
A-series stack against it on real THINGS-EEG (trial-averaged, sub-01).

For every comparison the SHIPPED attention projector is the single frozen judge
(so retrieval is measured exactly as the paper does). A1 range-coding is applied
to the shipped codecs too (their real on-disk size vs the shipped savez).

Outputs:
  * shipped_tiers  -- all 8 shipped codecs: modeled bpp, ratio, MSE, image+text
                      retrieval top-1/5/10, savez on-disk, range-coded on-disk.
  * stacked_curve  -- the stacked codec (A1+A2+A3+A5+A6) swept over rate.
  * tier_matched   -- shipped neurozip tier vs stacked at EQUIVALENT compression.
  * extremes       -- stacked at xlow / xxlow (beyond the shipped range): how
                      much decodable info survives.
  * variable_rate_one_model -- A5: one gained model spanning rates.

Run:  .venv/bin/python tests/exp_shipped.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_real as R                      # sets C.TINY = real config; brings helpers
import exp_common as C
from clip_proj import load_frozen_projector
from codec import EEGCodec
from exp_stacked import StackedCodec, train_stacked
from exp_real import (retr_both, project, savez_bpp, eval_stacked_full,
                      range_code_latents, _interp, N_DISK)

C_LAT, HIDDEN = R.C_LAT, R.HIDDEN
LAM_TASK = 3.0
STACK_LAMS = (0.001, 0.004, 0.012, 0.04, 0.12, 0.4)   # spans ~64x .. ~1000x
TASK_TIER_BPP = {"low": 0.111, "med": 0.190, "high": 0.222, "xhigh": 0.250}


def ck_dir():
    return Path.home() / ".neurozip" / "checkpoints"


def load_shipped_codec(name, dev):
    p = ck_dir() / f"{name}.pt"
    st = torch.load(p, weights_only=False, map_location="cpu")
    cfg = st.get("config", {})
    m = EEGCodec(c_lat=cfg.get("c_lat", 32), hidden=cfg.get("hidden", 128),
                 n_attn=cfg.get("n_attn", 0))
    m.load_state_dict(st["model"])
    return m.eval().to(dev)


@torch.no_grad()
def eval_shipped(codec, data, dev, proj):
    c_lat = codec.c_lat
    factor = (c_lat * 32) / (C.N_CH * C.N_T)
    embs, mse, bps, n, lats = [], 0.0, 0.0, 0, []
    for (x,) in C.iterate(data["eeg_test"], batch=128, shuffle=False, dev=dev):
        y = codec.encoder(x); yq = torch.round(y); xh = codec.decoder(yq)
        embs.append(proj(xh).cpu()); lats.append(yq.cpu())
        mse += F.mse_loss(xh, x, reduction="sum").item()
        bps += codec.prior.bits(yq).item() * x.shape[0]; n += x.shape[0]
    ret = retr_both(torch.cat(embs), data)
    bpp = bps / n * factor
    lat = torch.cat(lats)[:N_DISK].numpy().astype(np.int64)
    sz = savez_bpp(lat, N_DISK)
    rc = range_code_latents(lat, codec.prior.log_b.detach().cpu().numpy())
    rc_bpp = rc["actual_bits"] / (N_DISK * C.N_CH * C.N_T)
    return {"bpp_modeled": round(bpp, 4), "ratio_modeled": round(16 / bpp, 1),
            "mse": round(mse / (n * C.N_CH * C.N_T), 5), "retrieval": ret,
            "ratio_savez_shipped": round(16 / sz, 1),
            "ratio_rangecoded_A1": round(16 / rc_bpp, 1), "rc_roundtrip": rc["roundtrip_ok"]}


def train_stacked_fixed(data, dev, lam, proj, epochs=12, batch=64):
    """Full stack at one rate (single gain level) — precise rate for tier match."""
    m = StackedCodec(n_levels=1).to(dev).train()          # c_lat/hidden from C.TINY
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(epochs):
        for x, tgt in C.iterate(data["eeg_train"], data["tgt_train"], batch=batch,
                                shuffle=True, dev=dev):
            xh, by, bz = m(x, 0)
            bpp = m.total_bpp(by, bz, x.shape[0])
            task = (1 - F.cosine_similarity(proj(xh), F.normalize(tgt, dim=-1), dim=-1)).mean()
            loss = F.mse_loss(xh, x) + lam * bpp + LAM_TASK * task
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def main():
    C.seed_all(0)
    dev = C.device()
    C.banner(f"STACKED (all A-series) vs SHIPPED v4 checkpoints  device={dev}")

    # Use the bundle's norm stats so shipped codecs see the exact normalization
    # they were trained with.
    bstats = ck_dir() / "norm_stats.pt"
    if bstats.exists():
        shutil.copy(bstats, Path("data") / "norm_stats.pt")

    R.PROJ_CAP = 20000                    # only need train for the stacked codec
    data = R.load_real()
    proj = load_frozen_projector(ck_dir() / "clip_proj.pt", device=dev)

    results = {"config": {"judge": "SHIPPED attention projector", "c_lat": C_LAT,
                          "hidden": HIDDEN, "eval": "trial-averaged 80 reps",
                          "n_disk_epochs": N_DISK},
               "chance_top1": round(1 / data["n_concepts"], 4)}
    ceil = retr_both(project(proj, data["eeg_test"], dev), data)
    results["shipped_judge_ceiling"] = ceil
    print(f"[ceiling shipped judge] {ceil}", flush=True)

    scores = json.loads((ck_dir() / "scores.json").read_text()) if (ck_dir() / "scores.json").exists() else {}

    # ---------- shipped tiers ----------
    shipped = {}
    for fam in ("fidelity", "neurozip"):
        for tier in ("low", "med", "high", "xhigh"):
            name = f"{fam}_v4_{tier}"
            try:
                m = load_shipped_codec(name, dev)
                ev = eval_shipped(m, data, dev, proj)
                ev["scores_json_ref"] = {k: scores.get(name, {}).get(k) for k in
                                         ("bpp", "compression_ratio_vs_fp16", "mse")}
                shipped[name] = ev
                print(f"[shipped] {name}: bpp={ev['bpp_modeled']} ratio={ev['ratio_modeled']}x "
                      f"mse={ev['mse']} img@5={ev['retrieval']['image']['top5']} "
                      f"savez={ev['ratio_savez_shipped']}x A1={ev['ratio_rangecoded_A1']}x", flush=True)
            except Exception as e:
                shipped[name] = {"ERROR": str(e)}
                print(f"[shipped] {name} FAIL: {e}", flush=True)
    results["shipped_tiers"] = shipped
    Path("tests/results/SHIPPED_all.json").write_text(json.dumps(results, indent=1))

    # ---------- stacked (all A-series), rate sweep for tier matching ----------
    # light codec-training subsample
    perm = torch.randperm(data["eeg_train"].shape[0])[:16000]
    data["eeg_train"] = data["eeg_train"][perm].contiguous()
    data["tgt_train"] = data["tgt_train"][perm].contiguous()
    data["lab_train"] = data["lab_train"][perm].contiguous()

    stacked_curve = []
    for lam in STACK_LAMS:
        m = train_stacked_fixed(data, dev, lam, proj)
        ev = eval_stacked_full(m, data, dev, proj, 0)
        ev["lambda"] = lam
        ev["ratio_modeled"] = round(16 / ev["bpp_modeled"], 1)
        stacked_curve.append(ev)
        print(f"[stacked] lam={lam}: bpp={ev['bpp_modeled']} ratio_rc={ev['ratio_rangecoded']}x "
              f"mse={ev['mse']} img@5={ev['retrieval']['image']['top5']}", flush=True)
    results["stacked_curve"] = stacked_curve
    Path("tests/results/SHIPPED_all.json").write_text(json.dumps(results, indent=1))

    # ---------- tier-matched: shipped neurozip tier vs stacked at equal bpp ----------
    s_bpp = [s["bpp_modeled"] for s in stacked_curve]
    def stk(metric_fn):
        return [metric_fn(s) for s in stacked_curve]
    tier_matched = []
    for tier, tbpp in TASK_TIER_BPP.items():
        sh = shipped.get(f"neurozip_v4_{tier}", {})
        if "ERROR" in sh:
            continue
        within = min(s_bpp) <= tbpp <= max(s_bpp)
        row = {"tier": tier, "target_bpp": tbpp,
               "shipped_ratio_modeled": sh["ratio_modeled"],
               "shipped_ratio_savez": sh["ratio_savez_shipped"],
               "shipped_ratio_A1_rangecoded": sh["ratio_rangecoded_A1"],
               "shipped_mse": sh["mse"],
               "shipped_img_top5": sh["retrieval"]["image"]["top5"],
               "shipped_txt_top5": sh["retrieval"]["text"]["top5"],
               "stacked_extrapolated": not within}
        if within:
            row["stacked_ratio_rangecoded_at_equal_bpp"] = round(_interp(s_bpp, stk(lambda s: s["ratio_rangecoded"]), tbpp), 1)
            row["stacked_mse_at_equal_bpp"] = round(_interp(s_bpp, stk(lambda s: s["mse"]), tbpp), 5)
            row["stacked_img_top5_at_equal_bpp"] = round(_interp(s_bpp, stk(lambda s: s["retrieval"]["image"]["top5"]), tbpp), 4)
            row["stacked_txt_top5_at_equal_bpp"] = round(_interp(s_bpp, stk(lambda s: s["retrieval"]["text"]["top5"]), tbpp), 4)
        tier_matched.append(row)
    results["tier_matched"] = tier_matched

    # ---------- extremes: highest-compression stacked points ----------
    ext = sorted(stacked_curve, key=lambda s: s["ratio_rangecoded"], reverse=True)[:2]
    results["extremes_xlow_xxlow"] = [{
        "label": lab, "bpp_modeled": s["bpp_modeled"],
        "ratio_rangecoded": s["ratio_rangecoded"], "mse": s["mse"],
        "img_top1": s["retrieval"]["image"]["top1"], "img_top5": s["retrieval"]["image"]["top5"],
        "img_top10": s["retrieval"]["image"]["top10"], "txt_top5": s["retrieval"]["text"]["top5"],
    } for lab, s in zip(("xxlow", "xlow"), ext)]

    # ---------- A5: one variable-rate stacked model spanning rates ----------
    try:
        vlams = (0.004, 0.02, 0.1, 0.4)
        vm = train_stacked(data, dev, vlams, proj, epochs=14)
        vr = []
        for lvl in range(len(vlams)):
            ev = eval_stacked_full(vm, data, dev, proj, lvl)
            vr.append({"level": lvl, "bpp_modeled": ev["bpp_modeled"],
                       "ratio_rangecoded": ev["ratio_rangecoded"], "mse": ev["mse"],
                       "img_top5": ev["retrieval"]["image"]["top5"]})
        results["variable_rate_one_model"] = vr
    except Exception as e:
        results["variable_rate_one_model"] = {"ERROR": str(e)}

    results["stack_contents"] = ["A1 range-coding", "A2+A3 mixture hyperprior",
                                 "A5 variable-rate gains", "A6 cosine task"]
    Path("tests/results/SHIPPED_all.json").write_text(json.dumps(results, indent=1))
    print("RESULTS_JSON:" + json.dumps(results))
    print("[done]", flush=True)
    return results


if __name__ == "__main__":
    main()
