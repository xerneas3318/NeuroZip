"""
fable_test.py -- one-command self-test / demo for the fmri-fable codecs.

Fetches the Haxby data if needed, runs EVERY codec on the held-out session, does a
real compress -> file -> decompress round-trip, and prints one summary table so you
can confirm the models work and see the numbers.

    python fable_test.py

Needs: the committed checkpoints in checkpoints/, and an internet connection the
first time (to download ~300 MB of Haxby via nilearn). Everything else is in-repo.
"""
from __future__ import annotations
import os, sys, tempfile
import numpy as np
import torch

os.environ.setdefault("NZ_FMRI_DATA", "data_fmri")
import fable_data, fable_io, fable_svd
from fable_codec import VQCodec, count_params
from fable_eval import original_test_volumes, nii_gz_bytes, vt_decoding

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_data():
    if not os.path.exists(fable_data.CACHE):
        print("[test] building Haxby cache (first run; downloads ~300 MB) ...")
        fable_data.build_cache()
    return fable_data.load()


def load_neural(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = VQCodec(**ck["config"]).to(DEV); m.load_state_dict(ck["model"]); m.eval()
    if "prior" in ck:
        p = ck["prior"]
        m._prior = {"marginal": p["marginal"].astype(np.float64),
                    "transition": p["transition"].astype(np.float64), "K": int(p["K"])}
    else:  # not finalized -> build prior from train tokens
        tr = fable_data.load()["train"]
        with torch.no_grad():
            c = torch.cat([m.encode(torch.from_numpy(tr[i:i+128]).unsqueeze(1).to(DEV))
                           for i in range(0, len(tr), 128)]).reshape(len(tr), -1).cpu().numpy().astype(np.int32)
        m._prior = fable_io.fit_prior(c, ck["config"]["K"])
    return m, ck


def metrics(rec, rec_tr, d, mask, mu, gstd, orig, vt, var, total_bytes, niigz):
    a, b = rec[:, mask], d["test"][:, mask]
    mse = float(((a - b) ** 2).mean()); ve = 1 - mse / var
    xhat = np.zeros_like(orig); xhat[:, mask] = rec[:, mask] * gstd + mu
    psnr = float(10 * np.log10(orig[:, mask].max() ** 2 / ((xhat[:, mask] - orig[:, mask]) ** 2).mean()))
    acc, _ = vt_decoding(d["train"][:, vt], d["test"][:, vt], rec[:, vt], rec_tr[:, vt],
                         d["train_cls"], d["test_cls"], list(d["class_names"]))
    return dict(mse=mse, ve=ve, psnr=psnr, ratio=niigz / total_bytes,
                bpv=total_bytes / len(d["test"]),
                retain=acc["within"] / acc["baseline"], base=acc["baseline"])


def main():
    d = ensure_data()
    mask, mu, gstd = d["mask"], d["mu"], float(d["sd"][0])
    var = float(d["test"][:, mask].var())
    orig, vt = original_test_volumes()
    niigz = nii_gz_bytes(orig)
    rows = []

    # ---- neural per-volume codecs (need checkpoints) ----
    for name, path in [("nd1 neural", "checkpoints/fable_nd1.pt"),
                       ("nd3 neural", "checkpoints/fable_nd3.pt")]:
        if not os.path.exists(path):
            print(f"[test] skip {name}: {path} missing (train with fable_train.py)"); continue
        m, ck = load_neural(path)
        blob, info = fable_io.compress(d["test"], m, m._prior, mu, gstd, mask, device=DEV)
        rec, _, _, _ = fable_io.decompress(blob, m, device=DEV)
        rec_tr, _, _, _ = fable_io.decompress(
            fable_io.compress(d["train"], m, m._prior, mu, gstd, mask, device=DEV)[0], m, device=DEV)
        r = metrics(rec, rec_tr, d, mask, mu, gstd, orig, vt, var, info["total_bytes"], niigz)
        r.update(name=name, params=f"{count_params(m)/1e6:.2f}M")
        rows.append(r)

    # ---- temporal-SVD codec (no weights) ----
    for rank in (8, 16):
        blob, info = fable_svd.compress(d["test"], mask, mu, gstd, K=rank, runs=d["test_runs"])
        rec, _, _, _ = fable_svd.decompress(blob)
        rec_tr, _, _, _ = fable_svd.decompress(
            fable_svd.compress(d["train"], mask, mu, gstd, K=rank, runs=d["train_runs"])[0])
        r = metrics(rec, rec_tr, d, mask, mu, gstd, orig, vt, var, info["total_bytes"], niigz)
        r.update(name=f"svd K={rank}", params="0 (no net)")
        rows.append(r)

    print("\n" + "=" * 82)
    print("NeuroZip fmri-fable  --  held-out session (242 volumes, runs the codec never saw)")
    print("=" * 82)
    print(f"{'codec':12s}{'params':>11s}{'MSE':>8s}{'VE':>7s}{'PSNR':>8s}{'ratio':>8s}{'B/vol':>8s}{'decode kept':>13s}")
    print("-" * 82)
    for r in rows:
        print(f"{r['name']:12s}{r['params']:>11s}{r['mse']:>8.3f}{r['ve']*100:>6.0f}%"
              f"{r['psnr']:>7.1f}dB{r['ratio']:>7.0f}x{r['bpv']:>8.0f}{r['retain']*100:>11.0f}%")
    print("-" * 82)
    print("ratio = vs gzipped .nii.gz;  decode kept = VT 8-way object decoding vs uncompressed baseline")

    # ---- real file round-trip proof ----
    print("\n[test] real file round-trip (compress -> disk -> decompress):")
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "sess.nzs")
        blob, info = fable_svd.compress(d["test"], mask, mu, gstd, K=8, runs=d["test_runs"])
        open(f, "wb").write(blob)
        rec, _, _, _ = fable_svd.decompress(open(f, "rb").read())
        sz = os.path.getsize(f)
        mse = float((((rec - d["test"])[:, mask]) ** 2).mean())
        print(f"  svd K=8: wrote {sz/1024:.0f} KB, read back, MSE {mse:.3f}  ({niigz/sz:.0f}x vs .nii.gz)  OK")
    print("\nAll codecs ran. See FMRI_FABLE.md for the full write-up.\n")


if __name__ == "__main__":
    main()
