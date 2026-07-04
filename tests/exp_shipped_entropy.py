"""Apply the stack's ENTROPY side (A1 range-coding + A2/A3 hyperprior/mixture)
to the SHIPPED v4 codecs' own latents.

This is the clean "equivalent compression" comparison: the codec is the shipped
checkpoint (so MSE and retrieval are identical to shipped — truly equivalent),
and we only change how the integer latents are serialized. For each shipped tier
we report the real on-disk ratio under:
  * savez_compressed   (what the shipped .nz uses today)
  * A1 range-coded     (factorized prior the codec already carries)
  * A1 + hyperprior     (A2, fit on that codec's train latents)
  * A1 + mixture        (A3, fit on that codec's train latents)

Run:  .venv/bin/python tests/exp_shipped_entropy.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_real as R
import exp_common as C
from exp_priors import Hyperprior, MixturePrior, fit_rate_model, eval_bits_per_symbol, range_code_latents
from exp_shipped import load_shipped_codec, ck_dir

N_DISK = 64
N_FIT = 8000                      # train epochs used to fit hyper/mixture


@torch.no_grad()
def latents(codec, eeg, dev, batch=128):
    out = []
    for (x,) in C.iterate(eeg, batch=batch, shuffle=False, dev=dev):
        out.append(torch.round(codec.encoder(x)).cpu())
    return torch.cat(out)


def main():
    C.seed_all(0)
    dev = C.device()
    C.banner("Stacked ENTROPY applied to SHIPPED v4 latents (equivalent compression)")
    R.PROJ_CAP = N_FIT
    data = R.load_real()
    tr = data["eeg_train"][:N_FIT]
    te = data["eeg_test"]

    rows = {}
    for fam in ("neurozip", "fidelity"):
        for tier in ("low", "med", "high", "xhigh"):
            name = f"{fam}_v4_{tier}"
            codec = load_shipped_codec(name, dev)
            c_lat = codec.c_lat
            lat_tr = latents(codec, tr, dev)
            lat_te = latents(codec, te, dev)
            factor = (c_lat * 32) / (C.N_CH * C.N_T)

            with torch.no_grad():
                fact_bps = float(codec.prior.bits(lat_te.to(dev)).item())
            # A1 real on disk (factorized) on N_DISK test epochs
            sub = lat_te[:N_DISK].numpy().astype(np.int64)
            buf = io.BytesIO(); np.savez_compressed(buf, latents=sub.astype(np.int16))
            savez_bpp = buf.getbuffer().nbytes * 8 / (N_DISK * C.N_CH * C.N_T)
            rc = range_code_latents(sub, codec.prior.log_b.detach().cpu().numpy())
            a1_bpp = rc["actual_bits"] / (N_DISK * C.N_CH * C.N_T)
            # A2/A3 modeled bits on test latents (fit on train latents)
            hyper = fit_rate_model(Hyperprior(c_lat=c_lat), lat_tr, dev, epochs=30)
            mix = fit_rate_model(MixturePrior(c_lat=c_lat, k=4), lat_tr, dev, epochs=50)
            hyper_bps = eval_bits_per_symbol(hyper, lat_te, dev)
            mix_bps = eval_bits_per_symbol(mix, lat_te, dev)

            rows[name] = {
                "modeled_bpp_factorized": round(fact_bps * factor, 4),
                "ratio_savez_shipped": round(16 / savez_bpp, 1),
                "ratio_A1_rangecoded": round(16 / a1_bpp, 1),
                "A1_over_savez_x": round((16 / a1_bpp) / (16 / savez_bpp), 3),
                "bits_per_symbol_factorized": round(fact_bps, 4),
                "bits_per_symbol_hyperprior": round(hyper_bps, 4),
                "bits_per_symbol_mixture": round(mix_bps, 4),
                "hyperprior_rate_reduction_pct": round(100 * (1 - hyper_bps / fact_bps), 2),
                "mixture_rate_reduction_pct": round(100 * (1 - mix_bps / fact_bps), 2),
                "rc_roundtrip": rc["roundtrip_ok"],
            }
            print(f"[{name}] savez={rows[name]['ratio_savez_shipped']}x "
                  f"A1={rows[name]['ratio_A1_rangecoded']}x (x{rows[name]['A1_over_savez_x']}) "
                  f"hyper={rows[name]['hyperprior_rate_reduction_pct']}% "
                  f"mix={rows[name]['mixture_rate_reduction_pct']}%", flush=True)
            Path("tests/results/SHIPPED_entropy.json").write_text(json.dumps(rows, indent=1))

    print("RESULTS_JSON:" + json.dumps(rows))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
