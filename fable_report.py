"""
fable_report.py -- rate / distortion / function curve across codec settings.

Evaluates every checkpoint, prints one comparison table, and saves a plot of
compression-ratio vs (variance-explained, VT decoding retention). This is the
honest summary: how the three axes trade off as the latent grid coarsens.

    python fable_report.py checkpoints/fable_nd1.pt checkpoints/fable_nd2.pt checkpoints/fable_nd3.pt
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
import fable_data, fable_io
from fable_eval import (load_codec, encode_all, original_test_volumes,
                        nii_gz_bytes, pca_ceiling, vt_decoding)


def eval_one(path, d, orig_test, vt):
    codec, ck = load_codec(path)
    K = ck["config"]["K"]
    mask, mu, gstd = d["mask"], ck["meta"]["mu"], float(ck["meta"]["sd"][0])
    tr, te = d["train"], d["test"]
    codec._prior = fable_io.fit_prior(encode_all(codec, tr), K)

    blob, info = fable_io.compress(te, codec, codec._prior, mu, gstd, mask)
    rec, _, _, _ = fable_io.decompress(blob, codec)
    rec_tr, _, _, _ = fable_io.decompress(fable_io.compress(tr, codec, codec._prior, mu, gstd, mask)[0], codec)

    a, b = rec[:, mask], te[:, mask]
    ve = 1 - ((a - b)**2).mean() / b.var()
    av, bv = a - a.mean(0), b - b.mean(0)
    corr = float(((av*bv).mean(0) / (av.std(0)*bv.std(0) + 1e-8)).mean())
    xhat = np.zeros_like(orig_test); xhat[:, mask] = rec[:, mask]*gstd + mu
    psnr = float(10*np.log10(orig_test[:, mask].max()**2 / ((xhat[:, mask]-orig_test[:, mask])**2).mean()))

    acc, _ = vt_decoding(tr[:, vt], te[:, vt], rec[:, vt], rec_tr[:, vt],
                         d["train_cls"], d["test_cls"], list(d["class_names"]))
    niigz = nii_gz_bytes(orig_test)
    from fable_codec import count_params
    return dict(name=path.split("/")[-1].replace(".pt", ""),
                params_M=count_params(codec)/1e6, tokens=codec.tokens_per_vol,
                grid=tuple(codec.grid), K=K,
                total_B=info["total_bytes"], token_B=info["token_bytes"],
                ratio=niigz/info["total_bytes"], ratio_tok=niigz/info["token_bytes"],
                Bpv=info["total_bytes"]/info["n_vols"], ve=ve, corr=corr, psnr=psnr,
                dec_base=acc["baseline"], dec_arch=acc["archival"], dec_within=acc["within"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--json", default="plots/fable_report.json")
    ap.add_argument("--plot", default="plots/fable_rd.png")
    args = ap.parse_args()
    d = fable_data.load()
    orig_test, vt = original_test_volumes()
    rows = [eval_one(p, d, orig_test, vt) for p in args.ckpts]
    rows.sort(key=lambda r: r["tokens"])

    print("="*94)
    print(f"{'codec':10s}{'params':>8s}{'tokens':>8s}{'B/vol':>8s}{'ratio':>9s}"
          f"{'VE%':>7s}{'corr':>7s}{'PSNR':>7s}{'dec_base':>9s}{'dec_dec':>8s}{'retain':>8s}")
    print("-"*94)
    for r in rows:
        print(f"{r['name']:10s}{r['params_M']:7.2f}M{r['tokens']:8d}{r['Bpv']:8.0f}"
              f"{r['ratio']:8.0f}x{r['ve']*100:7.1f}{r['corr']:7.3f}{r['psnr']:7.1f}"
              f"{r['dec_base']*100:8.1f}%{r['dec_within']*100:7.1f}%"
              f"{r['dec_within']/r['dec_base']*100:7.0f}%")
    print("="*94)
    print("ratio = vs gzipped .nii.gz (held-out session, self-contained artifact)")
    print("dec_* = VT 8-way object decoding (chance 12.5%); retain = within-decompressed / baseline")

    json.dump(rows, open(args.json, "w"), indent=2, default=float)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        rt = [r["ratio"] for r in rows]
        ax[0].plot(rt, [r["ve"]*100 for r in rows], "o-", color="#2a6f97")
        for r in rows: ax[0].annotate(r["name"], (r["ratio"], r["ve"]*100), fontsize=8)
        ax[0].set(xscale="log", xlabel="compression vs .nii.gz (log)", ylabel="variance explained (%)",
                  title="Rate–distortion")
        ax[0].grid(alpha=.3)
        ax[1].plot(rt, [r["dec_within"]/r["dec_base"]*100 for r in rows], "s-", color="#b23b34")
        for r in rows: ax[1].annotate(r["name"], (r["ratio"], r["dec_within"]/r["dec_base"]*100), fontsize=8)
        ax[1].axhline(100, ls="--", color="gray", lw=1, label="lossless-decoding")
        ax[1].set(xscale="log", xlabel="compression vs .nii.gz (log)", ylabel="VT decoding retained (%)",
                  title="Rate–function"); ax[1].grid(alpha=.3); ax[1].legend()
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print(f"plot -> {args.plot}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
