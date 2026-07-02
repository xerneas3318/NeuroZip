"""
redundancy_spectrum.py -- the paper's organizing analysis.

Thesis: a signal compresses exactly as much as it is redundant, and *where* the
redundancy lives (across channels/space vs across time) dictates which codec wins.
We measure this consistently across modalities and a max-entropy control.

For each modality we take many (units x time) samples and compute, from the
population covariances:
  - temporal redundancy  = 1 - (PCs for 90% of the TIME covariance) / T
  - spatial  redundancy  = 1 - (PCs for 90% of the UNIT covariance) / C
  - lag-1 temporal autocorrelation
  - intrinsic dimensionality = PCs for 90% of the flattened C*T space / (C*T)
Then we place each modality on the spectrum and relate it to the codec's achieved
compression ratio.

    python redundancy_spectrum.py
"""
from __future__ import annotations
import numpy as np
import torch

# Codec compression ratios at a high-fidelity operating point (VE ~>=90%),
# from this repo's branches: ECG/fMRI measured here, EEG from the EEG codec.
RATIOS = {"ECG": 30, "EEG": 70, "fMRI": 41, "random": 1}


def redundancy(cov):
    """1 - participation_ratio/dim. White noise -> 0; low-rank -> ->1. Scale-free."""
    ev = np.linalg.eigvalsh(cov); ev = ev[ev > 1e-12]
    pr = (ev.sum() ** 2) / (ev ** 2).sum()
    return 1 - pr / len(ev)


def diagnose(X, name):
    """X: (n, C, T) samples (each unit's time series centered)."""
    X = X.astype(np.float64)
    X = X - X.mean(2, keepdims=True)
    n, C, T = X.shape
    Ct = np.tensordot(X, X, axes=([0, 1], [0, 1])) / (n * C)      # (T,T) temporal cov
    Xs = np.transpose(X, (0, 2, 1)).reshape(n * T, C)
    Cs = (Xs.T @ Xs) / (n * T)                                   # (C,C) spatial cov
    a, b = X[:, :, 1:], X[:, :, :-1]
    ac = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
    R = np.corrcoef(Xs.T)                                        # inter-channel corr
    scorr = float(np.abs(R[np.triu_indices(C, 1)]).mean())
    return dict(name=name, C=C, T=T,
                temporal_red=redundancy(Ct), spatial_red=redundancy(Cs),
                autocorr=ac, spatial_corr=scorr)


def load_eeg(n=2000):
    d = torch.load("data_eeg/Preprocessed_data_250Hz_whiten/sub-01/test.pt",
                   map_location="cpu", weights_only=False)
    e = np.asarray(d["eeg"]).reshape(-1, 63, 250)
    rng = np.random.default_rng(0)
    return e[rng.choice(len(e), min(n, len(e)), replace=False)]


def load_ecg(n=2000):
    d = np.load("data_ecg/ecg_fable_cache.npz", allow_pickle=True)
    r = d["recs"]                     # (N,12,1000)
    w = r[:, :, :1000].reshape(len(r), 12, 4, 250).transpose(0, 2, 1, 3).reshape(-1, 12, 250)
    rng = np.random.default_rng(0)
    return w[rng.choice(len(w), min(n, len(w)), replace=False)]


def load_fmri(nvox=500):
    d = np.load("data_fmri/fable_cache.npz", allow_pickle=True)
    mask = d["mask"]; runs = d["train_runs"]
    vols = d["train"]                 # (N,40,56,48)
    V = vols[:, mask]                 # (N_time, nvox)
    rng = np.random.default_rng(0)
    vi = rng.choice(V.shape[1], nvox, replace=False)
    # one (nvox x T) sample per run
    samples = []
    for r in np.unique(runs):
        seg = V[runs == r][:, vi].T           # (nvox, T_run)
        T = 120
        if seg.shape[1] >= T:
            samples.append(seg[:, :T])
    return np.stack(samples)                  # (n_runs, nvox, 120)


def load_random(n=2000, C=63, T=250):
    return np.random.default_rng(0).standard_normal((n, C, T)).astype(np.float32)


def main():
    mods = []
    print("loading modalities ...")
    mods.append(diagnose(load_ecg(), "ECG"))
    mods.append(diagnose(load_eeg(), "EEG"))
    mods.append(diagnose(load_fmri(), "fMRI"))
    mods.append(diagnose(load_random(), "random"))

    print("\n" + "=" * 84)
    print("REDUNDANCY SPECTRUM  (higher redundancy = more compressible)")
    print("=" * 84)
    print(f"{'modality':9s}{'shape':>12s}{'temporal_red':>14s}{'spatial_red':>13s}"
          f"{'autocorr':>10s}{'chan_corr':>11s}{'codec':>8s}")
    print("-" * 84)
    for m in sorted(mods, key=lambda x: -RATIOS[x["name"]]):
        shape = f"{m['C']}x{m['T']}"
        print(f"{m['name']:9s}{shape:>12s}"
              f"{m['temporal_red']*100:>13.0f}%{m['spatial_red']*100:>12.0f}%"
              f"{m['autocorr']:>10.2f}{m['spatial_corr']:>11.2f}{RATIOS[m['name']]:>7d}x")
    print("-" * 84)
    print("temporal_red / spatial_red = 1 - participation_ratio/dim  (0 = white noise, ->1 = low-rank)")
    print("autocorr = lag-1 temporal; chan_corr = mean |inter-channel correlation|")
    print("codec = compression ratio at high fidelity (ECG/fMRI measured here, EEG from EEG codec)")

    import json
    json.dump([{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in m.items()} for m in mods],
              open("plots/redundancy_spectrum.json", "w"), indent=2)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"ECG": "#b23b34", "EEG": "#2a6f97", "fMRI": "#4a7c59", "random": "#888"}
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        for m in mods:
            ax[0].scatter(m["autocorr"], m["spatial_corr"], s=150, color=col[m["name"]])
            ax[0].annotate(m["name"], (m["autocorr"], m["spatial_corr"]),
                           textcoords="offset points", xytext=(7, 6), fontsize=11)
        ax[0].set(xlabel="temporal redundancy  (lag-1 autocorrelation)",
                  ylabel="channel redundancy  (mean |inter-channel corr|)",
                  title="Where the redundancy lives")
        ax[0].grid(alpha=.3)
        for m in mods:
            ax[1].scatter(m["autocorr"], RATIOS[m["name"]], s=150, color=col[m["name"]])
            ax[1].annotate(m["name"], (m["autocorr"], RATIOS[m["name"]]),
                           textcoords="offset points", xytext=(7, 6), fontsize=11)
        ax[1].set(xlabel="temporal redundancy (lag-1 autocorrelation)", ylabel="codec compression (x)",
                  yscale="log", title="Compressibility tracks redundancy")
        ax[1].grid(alpha=.3, which="both")
        fig.tight_layout(); fig.savefig("plots/redundancy_spectrum.png", dpi=130)
        print("\nplot -> plots/redundancy_spectrum.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    import os; os.makedirs("plots", exist_ok=True)
    main()
