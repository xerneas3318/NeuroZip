"""
fable_svd.py -- temporal-transform codec for 4D fMRI (the low-MSE codec).

Diagnostic: an fMRI run is dramatically low-rank in TIME (rank-16 temporal SVD of
121 TRs -> MSE 0.056) because the stimulus is a block design and BOLD is slow --
whereas each single volume is high-rank in SPACE (per-volume PCA floors ~0.14). So
the per-volume neural codec (fable_codec.py), which compresses each volume's
spatial pattern independently, floors around MSE 0.24 no matter how it's tuned.

This codec compresses along the axis that is actually redundant. Per temporal
SEGMENT it takes an SVD of the (T x voxels) matrix, keeps the top-K components,
and stores K full-resolution spatial maps (int8) + tiny per-frame temporal
coefficients (float16). Full spatial resolution is retained (that's why MSE goes
low); only the time axis is truncated. Classic transform coding (KLT), matched to
the signal. No training -- the basis is computed per segment at compress time.

    python fable_svd.py --rank 8            # MSE ~0.09 @ ~50x
    python fable_svd.py --rank 16           # MSE ~0.07 @ ~27x
"""
from __future__ import annotations
import argparse, struct, gzip
import numpy as np

MAGIC = b"NZS1"


def _pack_mask(mask):
    return np.packbits(mask.reshape(-1).astype(np.uint8)).tobytes()


def _unpack_mask(blob, shape):
    n = int(np.prod(shape))
    return np.unpackbits(np.frombuffer(blob, np.uint8), count=n).reshape(shape).astype(bool)


def segments(n, seg_len, runs=None):
    """Yield (start, stop) blocks. Respect run boundaries if given, else fixed seg_len."""
    if runs is not None:
        for r in np.unique(runs):
            idx = np.where(runs == r)[0]
            s0, s1 = int(idx[0]), int(idx[-1]) + 1
            for s in range(s0, s1, seg_len):
                yield s, min(s + seg_len, s1)
    else:
        for s in range(0, n, seg_len):
            yield s, min(s + seg_len, n)


def compress(vols_norm, mask, mu, gstd, K=8, seg_len=121, runs=None):
    """vols_norm: (T,D,H,W) normalized fluctuation (background 0). Returns (blob, info)."""
    D, H, W = mask.shape
    X = vols_norm[:, mask]                                   # (T, nvox)
    T, nvox = X.shape
    body = bytearray()
    n_seg = 0
    for s, e in segments(T, seg_len, runs):
        M = X[s:e]                                           # (t, nvox)
        k = min(K, M.shape[0])
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        A = (U[:, :k] * S[:k]).astype(np.float16)            # (t, k) temporal coeffs
        Vk = Vt[:k]                                          # (k, nvox) spatial maps
        scale = (np.abs(Vk).max(1) + 1e-8).astype(np.float32)   # per-map int8 scale
        Vq = np.round(Vk / scale[:, None] * 127).astype(np.int8)
        body += struct.pack("<HH", e - s, k)
        body += A.tobytes() + scale.tobytes() + Vq.tobytes()
        n_seg += 1
    mu16 = mu.astype(np.float16).tobytes()
    mask_b = _pack_mask(mask)
    header = MAGIC + struct.pack("<IHHHHHIf", T, D, H, W, K, seg_len, len(mask_b), float(gstd))
    header += struct.pack("<I", len(mu16))
    raw = header + mask_b + mu16 + struct.pack("<I", len(body)) + bytes(body)
    blob = gzip.compress(raw, 6)                             # maps/coeffs still have structure
    info = dict(n_vols=T, n_segments=n_seg, rank=K, total_bytes=len(blob),
                map_bytes=len(body), Bpv=len(blob) / T)
    return blob, info


def decompress(blob):
    raw = gzip.decompress(blob)
    assert raw[:4] == MAGIC
    off = 4
    T, D, H, W, K, seg_len, n_mask, gstd = struct.unpack_from("<IHHHHHIf", raw, off)
    off += struct.calcsize("<IHHHHHIf")
    (n_mu,) = struct.unpack_from("<I", raw, off); off += 4
    mask = _unpack_mask(raw[off:off + n_mask], (D, H, W)); off += n_mask
    mu = np.frombuffer(raw[off:off + n_mu], np.float16).astype(np.float32); off += n_mu
    (n_body,) = struct.unpack_from("<I", raw, off); off += 4
    body = raw[off:off + n_body]; bo = 0
    nvox = int(mask.sum())
    out = np.zeros((T, D, H, W), np.float32)
    flat = np.zeros((T, nvox), np.float32)
    t = 0
    while bo < len(body):
        seg_T, k = struct.unpack_from("<HH", body, bo); bo += 4
        A = np.frombuffer(body[bo:bo + seg_T * k * 2], np.float16).astype(np.float32).reshape(seg_T, k); bo += seg_T * k * 2
        scale = np.frombuffer(body[bo:bo + k * 4], np.float32); bo += k * 4
        Vq = np.frombuffer(body[bo:bo + k * nvox], np.int8).astype(np.float32).reshape(k, nvox); bo += k * nvox
        Vd = Vq * scale[:, None] / 127
        flat[t:t + seg_T] = A @ Vd
        t += seg_T
    out[:, mask] = flat
    return out, mu, gstd, mask


def _eval(rank):
    import fable_data
    from fable_eval import original_test_volumes, nii_gz_bytes, vt_decoding
    d = fable_data.load()
    mask, mu, gstd = d["mask"], d["mu"], float(d["sd"][0])
    te, runs = d["test"], d["test_runs"]
    var = float(te[:, mask].var())

    blob, info = compress(te, mask, mu, gstd, K=rank, runs=runs)
    rec, mu2, gstd2, mask2 = decompress(blob)
    # lossless container check: decompress reproduces the codec's own reconstruction
    a, b = rec[:, mask], te[:, mask]
    mse = ((a - b) ** 2).mean(); ve = 1 - mse / var
    av, bv = a - a.mean(0), b - b.mean(0)
    corr = float(((av * bv).mean(0) / (av.std(0) * bv.std(0) + 1e-8)).mean())

    orig, vt = original_test_volumes()
    xhat = np.zeros_like(orig); xhat[:, mask] = rec[:, mask] * gstd + mu
    psnr = float(10 * np.log10(orig[:, mask].max() ** 2 / ((xhat[:, mask] - orig[:, mask]) ** 2).mean()))
    niigz = nii_gz_bytes(orig)

    rec_tr, _, _, _ = decompress(compress(d["train"], mask, mu, gstd, K=rank, runs=d["train_runs"])[0])
    acc, _ = vt_decoding(d["train"][:, vt], te[:, vt], rec[:, vt], rec_tr[:, vt],
                         d["train_cls"], d["test_cls"], list(d["class_names"]))
    print("=" * 60)
    print(f"TEMPORAL-SVD transform codec  |  rank K={rank}")
    print("=" * 60)
    print(f"  held-out session : {info['n_vols']} vols, {info['n_segments']} segments")
    print("-" * 60)
    print(f"  MSE (in-brain)   : {mse:.4f}   <-- target < 0.1")
    print(f"  variance expl.   : {ve*100:.1f}%")
    print(f"  per-voxel corr   : {corr:.3f}")
    print(f"  PSNR (orig units): {psnr:.1f} dB")
    print("-" * 60)
    print(f"  .nzs artifact    : {info['total_bytes']/1024:.0f} KB ({info['Bpv']:.0f} B/vol)")
    print(f"  compression      : {niigz/info['total_bytes']:.0f}x vs gzipped .nii.gz")
    print("-" * 60)
    print(f"  VT decoding      : base {acc['baseline']*100:.1f}% -> decomp {acc['within']*100:.1f}% "
          f"(retain {acc['within']/acc['baseline']*100:.0f}%)")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    import os
    os.environ.setdefault("NZ_FMRI_DATA", "data_fmri")
    _eval(args.rank)
