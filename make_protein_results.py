"""
make_protein_results.py — concrete reconstruction demo for the protein v4 codec.

For a few held-out proteins: compress -> reconstruct, show the original vs
reconstructed amino-acid sequence and a (20 x L) one-hot heatmap pair, with the
per-residue accuracy. The protein analog of the EEG reconstruction viewer.
"""
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protein_data import load_proteins, to_onehot, PAD, AA_ORDER
from protein_codec import ProteinCodec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/protein_lr0.5.pt")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="results/protein_reconstruction.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, weights_only=False, map_location=device)
    c = ck["config"]
    m = ProteinCodec(n_aa=c["n_aa"], c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    fv = ck["final_val"]

    _, (va_idx, va_lens), _ = load_proteins()
    rows = [i for i in range(args.n)]
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 2.3 * len(rows)))
    for r, ri in enumerate(rows):
        idx = torch.from_numpy(va_idx[ri:ri+1]); L = int(va_lens[ri])
        x = to_onehot(idx, device)
        with torch.no_grad():
            logits, _ = m.compress_then_reconstruct(x)
        pred = logits.argmax(1)[0].cpu().numpy()
        gold = idx[0].cpu().numpy()
        mask = gold != PAD
        acc = float((pred[mask] == gold[mask]).mean())
        orig_seq = "".join(AA_ORDER[a] for a in gold[:L])
        rec_seq = "".join(AA_ORDER[a] if a < 20 else "-" for a in pred[:L])
        xo = x[0].cpu().numpy()[:, :L]
        xr = np.zeros_like(xo)
        for p in range(L):
            xr[pred[p], p] = 1 if pred[p] < 20 else 0
        axes[r, 0].imshow(xo, aspect="auto", cmap="Greens", interpolation="nearest")
        axes[r, 0].set_title(f"original protein  (len {L})", fontsize=9)
        axes[r, 0].set_yticks(range(20)); axes[r, 0].set_yticklabels(list(AA_ORDER), fontsize=5)
        axes[r, 1].imshow(xr, aspect="auto", cmap="Greens", interpolation="nearest")
        axes[r, 1].set_title(f"reconstruction  —  {acc*100:.1f}% residues correct", fontsize=9)
        axes[r, 1].set_yticks(range(20)); axes[r, 1].set_yticklabels(list(AA_ORDER), fontsize=5)
        for a in axes[r]:
            a.set_xlabel("residue", fontsize=8)
        print(f"protein {r+1} (len {L}, acc {acc*100:.1f}%)")
        print(f"  orig: {orig_seq[:70]}{'...' if L>70 else ''}")
        print(f"  recon:{rec_seq[:70]}{'...' if L>70 else ''}")
    fig.suptitle(f"v4 codec on proteins — {fv['ratio']:.0f}× compression, {fv['acc']*100:.1f}% "
                 f"per-residue accuracy (same architecture as EEG)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    import os; os.makedirs("results", exist_ok=True); fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
