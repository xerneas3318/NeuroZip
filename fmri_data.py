"""
fmri_data.py — load ABIDE CC200 parcellated fMRI time series into (200 x 128)
windows, analogous to the THINGS-EEG loader.

Each subject's ROI time series (T TRs x 200 regions) is split into 128-TR
non-overlapping windows -> (200, 128) epochs, the fMRI analog of the (63, 250)
EEG epoch. Per-ROI z-score normalization (stats from train).

Run standalone for a summary:
    python fmri_data.py
"""

import glob
import os
import numpy as np

from fmri_codec import N_ROI, N_TIMES

ROOT = "data_fmri/rois"
CACHE = "data_fmri/fmri_windows.npz"


def build_cache(win=N_TIMES, stride=N_TIMES // 2, cache=CACHE):   # overlap -> more windows
    files = sorted(glob.glob(f"{ROOT}/*.1D"))
    wins, used = [], 0
    for f in files:
        try:
            ts = np.loadtxt(f, dtype=np.float32)          # (T, 200)
        except Exception:
            continue
        if ts.ndim != 2 or ts.shape[1] != N_ROI:
            continue
        x = ts.T                                          # (200, T)
        x = np.nan_to_num(x)
        T = x.shape[1]
        for s in range(0, T - win + 1, stride):
            w = x[:, s:s + win]
            if np.std(w) < 1e-6:
                continue
            wins.append(w)
        used += 1
    W = np.stack(wins).astype(np.float32)                 # (N, 200, 128)
    np.savez(cache, win=W)
    print(f"[fmri] built {len(W)} windows from {used} subjects -> {cache}")
    return W


def load_fmri(val_frac=0.05, seed=0, rebuild=False, clip=6.0, smooth_sigma=0.0):
    if rebuild or not os.path.exists(CACHE):
        W = build_cache()
    else:
        W = np.load(CACHE)["win"]
    rng = np.random.default_rng(seed)
    W = W[rng.permutation(len(W))]
    n_val = int(len(W) * val_frac)
    val, train = W[:n_val], W[n_val:]
    mean = train.mean((0, 2), keepdims=True)
    std = train.std((0, 2), keepdims=True) + 1e-6
    train = (train - mean) / std
    val = (val - mean) / std
    if clip:                       # winsorize motion spikes (standard fMRI scrubbing)
        train = np.clip(train, -clip, clip)
        val = np.clip(val, -clip, clip)
    if smooth_sigma:               # temporal low-pass: BOLD is <0.1 Hz, high-freq is noise
        from scipy.ndimage import gaussian_filter1d
        train = gaussian_filter1d(train, smooth_sigma, axis=2)
        val = gaussian_filter1d(val, smooth_sigma, axis=2)
    meta = {"n_train": len(train), "n_val": len(val), "n_roi": N_ROI, "n_times": N_TIMES,
            "mean": mean, "std": std}
    return train, val, meta


if __name__ == "__main__":
    train, val, meta = load_fmri()
    print("=" * 56)
    print("ABIDE CC200 fMRI — summary")
    print("=" * 56)
    print(f"  train / val windows : {meta['n_train']} / {meta['n_val']}")
    print(f"  ROIs (channels)     : {meta['n_roi']}  (CC200 atlas)")
    print(f"  TRs (time)          : {meta['n_times']}  (~{meta['n_times']*2}s @ TR≈2s)")
    print(f"  epoch shape         : (200, 128)   <- analog of EEG (63, 250)")
    print(f"  normalized range    : [{train.min():.2f}, {train.max():.2f}]  std≈{train.std():.2f}")
    print("=" * 56)
