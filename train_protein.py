"""
train_protein.py — train the v4 codec (conv-only, scalar quantize + Laplace) to
compress UniProt protein sequences. Demonstrates the EEG architecture transfers
to a new domain with NO structural change (just 63->20 input channels).

Reconstruction loss is cross-entropy over the 20 amino acids per residue (the
categorical analog of the EEG codec's MSE); rate is the same Laplace prior.
Reports per-residue reconstruction accuracy (chance = 5%), bits/residue, and
compression ratio vs the float16 one-hot input.

Run:
    python train_protein.py --epochs 25 --lambda-rate 0.02
"""

import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

from protein_data import load_proteins, to_onehot, PAD, N_AA, N_POS
from protein_codec import ProteinCodec


@torch.no_grad()
def evaluate(codec, idx_val, lens_val, device, bs=512):
    codec.eval()
    correct = total = 0
    bits_sum = nb = 0.0
    for i in range(0, idx_val.shape[0], bs):
        idx = idx_val[i:i+bs].to(device)
        x = to_onehot(idx, device)
        logits, bits = codec.compress_then_reconstruct(x)        # rounded latent
        pred = logits.argmax(1)                                  # (B, L)
        mask = idx != PAD
        correct += (pred[mask] == idx[mask]).sum().item()
        total += mask.sum().item()
        bits_sum += bits.item(); nb += 1
    acc = correct / total
    bps = bits_sum / nb
    bpp = codec.bpp_floor(bps)
    bits_per_epoch = bps * codec.c_lat * 32
    bits_per_residue = bits_per_epoch / float(lens_val.float().mean())
    return acc, bps, bpp, bits_per_residue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--c-lat", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-attn", type=int, default=0)            # v4 = conv-only
    ap.add_argument("--lambda-rate", type=float, default=0.02)
    ap.add_argument("--out", default="checkpoints/protein_codec.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    import os; os.makedirs("checkpoints", exist_ok=True)

    (tr_idx, tr_lens), (va_idx, va_lens), meta = load_proteins()
    tr_idx = torch.from_numpy(tr_idx); va_idx = torch.from_numpy(va_idx)
    va_lens = torch.from_numpy(va_lens.astype(np.float32))
    print(f"device={device}  train={meta['n_train']} val={meta['n_val']}  "
          f"(20 amino acids x 250 positions; chance acc=5.0%)")

    codec = ProteinCodec(n_aa=N_AA, c_lat=args.c_lat, hidden=args.hidden, n_attn=args.n_attn).to(device)
    print(f"v4 ProteinCodec c_lat={args.c_lat} hidden={args.hidden} n_attn={args.n_attn} "
          f"params={sum(p.numel() for p in codec.parameters())}")
    opt = torch.optim.Adam(codec.parameters(), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(0)
    N = tr_idx.shape[0]

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        codec.train(); t0 = time.time(); run = {"ce": 0.0, "nb": 0}
        order = torch.randperm(N, generator=gen)
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            idx = tr_idx[order[i:i+args.batch_size]].to(device)
            x = to_onehot(idx, device)
            logits, bits, _ = codec(x)
            target = idx.long().clone(); target[idx == PAD] = -100   # ignore padding
            ce = F.cross_entropy(logits, target, ignore_index=-100)
            loss = ce + args.lambda_rate * bits
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
            run["ce"] += ce.item(); run["nb"] += 1
        acc, bps, bpp, bpr = evaluate(codec, va_idx, va_lens, device)
        if acc > best:
            best = acc
            torch.save({"model": codec.state_dict(),
                        "config": {"c_lat": args.c_lat, "hidden": args.hidden, "n_attn": args.n_attn, "n_aa": N_AA},
                        "final_val": {"acc": acc, "bits_per_symbol": bps, "bpp": bpp,
                                      "bits_per_residue": bpr, "ratio": 16.0 / bpp}}, args.out)
        print(f"ep{epoch:2d}/{args.epochs} train_ce={run['ce']/run['nb']:.3f} | "
              f"per-residue acc={acc*100:5.1f}%  bits/residue={bpr:.2f}  "
              f"ratio={16.0/bpp:.0f}x vs fp16 ({time.time()-t0:.1f}s)")
    print(f"\nPROTEIN v4 codec: best per-residue acc={best*100:.1f}% (chance 5%)  -> {args.out}")


if __name__ == "__main__":
    main()
