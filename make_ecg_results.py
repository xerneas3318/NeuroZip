"""
make_ecg_results.py — ECG reconstruction demo: original vs reconstructed 12-lead
waveforms (the heartbeat the codec preserves), with per-epoch MSE.
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_data import load_ecg
from ecg_codec import ECGCodec, LEAD_NAMES, N_TIMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ecg_codec.pt")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="results/ecg_reconstruction.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, weights_only=False, map_location=device)
    c = ck["config"]
    m = ECGCodec(n_leads=c["n_leads"], c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    fv = ck["final_val"]
    _, val, _ = load_ecg()
    va = torch.from_numpy(val).to(device)

    t = np.linspace(0, 2.5, N_TIMES)
    fig, axes = plt.subplots(12, args.n, figsize=(6.2 * args.n, 11), sharex=True)
    for col in range(args.n):
        x = va[col:col+1]
        with torch.no_grad():
            xh, _ = m.compress_then_reconstruct(x)
        xo = x[0].cpu().numpy(); xr = xh[0].cpu().numpy()
        mse = float(((xo - xr) ** 2).mean())
        for lead in range(12):
            ax = axes[lead, col] if args.n > 1 else axes[lead]
            ax.plot(t, xo[lead], color="0.45", lw=1.0, label="original")
            ax.plot(t, xr[lead], color="#d62728", lw=1.0, alpha=0.85, label="reconstruction")
            ax.set_ylabel(LEAD_NAMES[lead], fontsize=8, rotation=0, ha="right", va="center")
            ax.set_yticks([])
            if lead == 0:
                ax.set_title(f"ECG #{col} — {mse:.3f} MSE", fontsize=10)
                if col == 0: ax.legend(fontsize=8, loc="upper right")
        (axes[-1, col] if args.n > 1 else axes[-1]).set_xlabel("time (s)")
        print(f"ECG {col}: MSE={mse:.4f}")
    fig.suptitle(f"v4 codec on ECG — {fv['ratio']:.0f}× compression, {fv['var_exp']*100:.0f}% variance "
                 f"explained (same architecture as EEG, 63→12 leads)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    import os; os.makedirs("results", exist_ok=True); fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
