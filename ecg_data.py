"""
ecg_data.py — load PTB-XL 12-lead ECG (100 Hz) into (12 x 250) epochs, analogous
to the THINGS-EEG loader.

Each 10 s record (12 leads x 1000 samples) is split into 250-sample windows
(2.5 s), giving (12, 250) epochs — the ECG analog of the (63, 250) EEG epoch.
Per-lead z-score normalization (stats from train), like the EEG codec.

Run standalone for a summary:
    python ecg_data.py
"""

import glob
import os
import numpy as np

from ecg_codec import N_LEADS, N_TIMES, LEAD_NAMES

ROOT = "data_ecg"
CACHE = "data_ecg/ecg_windows.npz"


def build_cache(max_records=4000, win=N_TIMES, stride=N_TIMES, cache=CACHE):
    import wfdb
    heas = sorted(glob.glob(f"{ROOT}/**/*.hea", recursive=True))
    if max_records:
        heas = heas[:max_records]
    wins = []
    used = 0
    for h in heas:
        rec = h[:-4]
        try:
            sig, _ = wfdb.rdsamp(rec)              # (1000, 12)
        except Exception:
            continue
        if sig.shape[1] != N_LEADS:
            continue
        x = np.nan_to_num(sig.T.astype(np.float32))  # (12, 1000)
        T = x.shape[1]
        for s in range(0, T - win + 1, stride):
            w = x[:, s:s + win]
            if np.abs(w).max() < 1e-6:             # skip flatline windows
                continue
            wins.append(w)
        used += 1
    W = np.stack(wins).astype(np.float32)          # (N, 12, 250)
    np.savez(cache, win=W)
    print(f"[ecg] built {len(W)} windows from {used} records -> {cache}")
    return W


def load_ecg(val_frac=0.05, seed=0, rebuild=False):
    if rebuild or not os.path.exists(CACHE):
        W = build_cache()
    else:
        W = np.load(CACHE)["win"]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(W))
    W = W[perm]
    n_val = int(len(W) * val_frac)
    val, train = W[:n_val], W[n_val:]
    # per-lead z-score from train
    mean = train.mean((0, 2), keepdims=True)
    std = train.std((0, 2), keepdims=True) + 1e-6
    train = (train - mean) / std
    val = (val - mean) / std
    meta = {"n_train": len(train), "n_val": len(val), "n_leads": N_LEADS, "n_times": N_TIMES,
            "mean": mean, "std": std, "lead_names": LEAD_NAMES}
    return train, val, meta


if __name__ == "__main__":
    train, val, meta = load_ecg()
    print("=" * 56)
    print("PTB-XL 12-lead ECG — summary")
    print("=" * 56)
    print(f"  train / val epochs : {meta['n_train']} / {meta['n_val']}")
    print(f"  leads (channels)   : {meta['n_leads']}  ({', '.join(LEAD_NAMES)})")
    print(f"  samples (time)     : {meta['n_times']}  (2.5 s @ 100 Hz)")
    print(f"  epoch shape        : (12, 250)   <- analog of EEG (63, 250)")
    print(f"  normalized range   : [{train.min():.2f}, {train.max():.2f}]  std≈{train.std():.2f}")
    print("=" * 56)
