"""
fable_eval.py -- end-to-end evaluation of the fMRI fable codec.

Reports, on the held-out test runs:
  1. RATE   -- real bytes of the .nzf artifact vs baselines (float32, int16,
               gzip/.nii.gz -- what a lab actually stores).
  2. FIDELITY -- variance-explained (in-brain), per-voxel temporal correlation,
               PSNR in original scanner units; plus the linear PCA ceiling at a
               matched number of latents.
  3. FUNCTION -- does the decodable signal survive? Ventral-temporal 8-way object
               decoding accuracy on ORIGINAL vs DECOMPRESSED volumes. This is the
               fMRI analog of NeuroZip's "keep the task-relevant content" thesis.

    python fable_eval.py --ckpt checkpoints/fable_nd3.pt
"""
from __future__ import annotations
import argparse, gzip, io, time
import numpy as np
import torch
import nibabel as nib
import fable_data, fable_io
from fable_codec import VQCodec, count_params

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_codec(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = VQCodec(**ck["config"]).to(DEV); m.load_state_dict(ck["model"]); m.eval()
    return m, ck


def encode_all(codec, vols):
    with torch.no_grad():
        c = torch.cat([codec.encode(torch.from_numpy(vols[i:i+128]).unsqueeze(1).to(DEV))
                       for i in range(0, len(vols), 128)])
    return c.reshape(len(vols), -1).cpu().numpy().astype(np.int32)


def original_test_volumes():
    """Reload raw Haxby bold, crop, split -> original-unit test volumes + labels."""
    from nilearn import datasets
    hx = datasets.fetch_haxby(subjects=1, data_dir=fable_data.DATA_DIR)
    import pandas as pd
    bold = nib.load(hx.func[0]).get_fdata(dtype=np.float32)[fable_data.CROP[0], fable_data.CROP[1], fable_data.CROP[2], :]
    lab = pd.read_csv(hx.session_target[0], sep=" ")
    runs = lab["chunks"].to_numpy()
    is_test = np.isin(runs, (10, 11))
    orig_test = np.moveaxis(bold[..., is_test], -1, 0)          # (T,D,H,W)
    vt = nib.load(hx.mask_vt[0]).get_fdata()[fable_data.CROP[0], fable_data.CROP[1], fable_data.CROP[2]] > 0
    return orig_test, vt


def nii_gz_bytes(vols_int16):
    """Bytes if stored as a standard gzipped NIfTI (what labs actually store)."""
    img = nib.Nifti1Image(np.moveaxis(vols_int16, 0, -1).astype(np.int16), affine=np.eye(4))
    return len(gzip.compress(img.to_bytes(), 6))          # gzip level 6 = .nii.gz default


def pca_ceiling(train, test, mask, n_latent):
    from sklearn.decomposition import PCA
    tr = train[:, mask]; te = test[:, mask]
    mu = tr.mean(0)
    p = PCA(n_components=min(n_latent, tr.shape[0]-1, tr.shape[1]),
            svd_solver="randomized", random_state=0).fit(tr - mu)
    rec = (te - mu) @ p.components_.T @ p.components_ + mu
    return 1 - ((rec - te)**2).mean() / te.var()


def _fit_decoder(feat, lbl):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(feat)
    clf = LogisticRegression(max_iter=2000, C=0.05).fit(sc.transform(feat), lbl)
    return sc, clf


def vt_decoding(tr_o, te_o, te_d, tr_d, train_lbl, test_lbl, class_names):
    """8-way VT object decoding. Three scenarios separate 'info destroyed' from
    'domain shift':
      baseline    : train original      -> test original
      archival    : train original      -> test decompressed   (practical: decode archived data)
      within-comp : train decompressed  -> test decompressed   (is the info still there at all?)
    """
    rest = list(class_names).index("rest")
    trm, tem = train_lbl != rest, test_lbl != rest
    sc, clf = _fit_decoder(tr_o[trm], train_lbl[trm])
    baseline = clf.score(sc.transform(te_o[tem]), test_lbl[tem])
    archival = clf.score(sc.transform(te_d[tem]), test_lbl[tem])
    scd, clfd = _fit_decoder(tr_d[trm], train_lbl[trm])
    within = clfd.score(scd.transform(te_d[tem]), test_lbl[tem])
    return dict(baseline=baseline, archival=archival, within=within), int(tem.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/fable_nd3.pt")
    args = ap.parse_args()

    codec, ck = load_codec(args.ckpt)
    K = ck["config"]["K"]
    d = fable_data.load()
    mask, mu, gstd = d["mask"], ck["meta"]["mu"], float(ck["meta"]["sd"][0])
    tr, te = d["train"], d["test"]
    cls_names = list(d["class_names"])

    # prior from train tokens
    codec._prior = fable_io.fit_prior(encode_all(codec, tr), K)

    # compress / decompress the held-out session
    t0 = time.time(); blob, info = fable_io.compress(te, codec, codec._prior, mu, gstd, mask)
    t_comp = time.time() - t0
    t0 = time.time(); rec, _, _, _ = fable_io.decompress(blob, codec)
    t_dec = time.time() - t0

    # ---- fidelity (normalized, in-brain) ----
    a, b = rec[:, mask], te[:, mask]
    ve = 1 - ((a - b)**2).mean() / b.var()
    # per-voxel temporal correlation
    av, bv = a - a.mean(0), b - b.mean(0)
    corr = ((av*bv).mean(0) / (av.std(0)*bv.std(0) + 1e-8)).mean()

    # ---- original-unit fidelity + baselines ----
    orig_test, vt = original_test_volumes()
    # rebuild original-unit reconstruction: x_hat = rec*gstd + mu (in-brain)
    xhat = np.zeros_like(orig_test)
    xhat[:, mask] = rec[:, mask] * gstd + mu
    peak = orig_test[:, mask].max()
    mse_orig = ((xhat[:, mask] - orig_test[:, mask])**2).mean()
    psnr = 10*np.log10(peak**2 / mse_orig)

    n = len(te)
    f32 = orig_test.astype(np.float32).nbytes
    i16 = orig_test.astype(np.int16).nbytes
    niigz = nii_gz_bytes(orig_test)
    art = info["total_bytes"]; tok = info["token_bytes"]

    print("="*66)
    print(f"NeuroZip fMRI-fable codec  |  {args.ckpt}")
    print("="*66)
    print(f"model            : {count_params(codec)/1e6:.2f}M params  grid {codec.grid} "
          f"= {codec.tokens_per_vol} tokens/vol  K={K}")
    print(f"held-out session : {n} volumes (runs 10-11), crop {mask.shape}, "
          f"{int(mask.sum())} in-brain voxels")
    print("-"*66)
    print("RATE (real bytes on disk)")
    print(f"  float32 raw           {f32/1e6:8.2f} MB")
    print(f"  int16  raw            {i16/1e6:8.2f} MB")
    print(f"  gzip int16 (.nii.gz)  {niigz/1e6:8.2f} MB   <- standard lab storage")
    print(f"  .nzf artifact         {art/1e6:8.3f} MB   ({art/n:.0f} B/vol)")
    print(f"     tokens {tok/1024:.1f} KB + baseline mu {info['mu_bytes']/1024:.0f} KB "
          f"+ mask {info['mask_bytes']/1024:.0f} KB")
    print(f"  COMPRESSION vs .nii.gz  {niigz/art:6.1f}x   (vs int16 {i16/art:.0f}x, vs float32 {f32/art:.0f}x)")
    print(f"  tokens-only (amortized codec): {niigz/tok:.0f}x vs .nii.gz")
    print("-"*66)
    print("FIDELITY (held-out)")
    print(f"  variance explained (in-brain) : {ve*100:5.1f}%")
    print(f"  per-voxel temporal corr       : {corr:5.3f}")
    print(f"  PSNR (original units)         : {psnr:5.1f} dB")
    for k in (32, 64, codec.tokens_per_vol):
        print(f"  [ref] linear PCA @ {k:4d} latents VE = {pca_ceiling(tr, te, mask, k)*100:.1f}%")
    print("-"*66)
    print("FUNCTION: ventral-temporal 8-way object decoding (does signal survive?)")
    # decompress the train session too, for within-representation decoding
    blob_tr, _ = fable_io.compress(tr, codec, codec._prior, mu, gstd, mask)
    rec_tr, _, _, _ = fable_io.decompress(blob_tr, codec)
    acc, ntest = vt_decoding(tr[:, vt], te[:, vt], rec[:, vt], rec_tr[:, vt],
                             d["train_cls"], d["test_cls"], cls_names)
    print(f"  VT voxels={int(vt.sum())}  test trials={ntest}  chance={100/8:.0f}%")
    print(f"  baseline    (orig->orig)      : {acc['baseline']*100:5.1f}%")
    print(f"  archival    (orig->decomp)    : {acc['archival']*100:5.1f}%   "
          f"(retention {acc['archival']/acc['baseline']*100:.0f}%)")
    print(f"  within-comp (decomp->decomp)  : {acc['within']*100:5.1f}%   "
          f"(retention {acc['within']/acc['baseline']*100:.0f}%)")
    print("-"*66)
    print(f"speed: compress {t_comp:.2f}s  decompress {t_dec:.2f}s  ({n} vols)")
    print("="*66)


if __name__ == "__main__":
    main()
