"""
train_ecg.py — train the v4 codec (conv-only, scalar quantize + Laplace) on
PTB-XL ECG. ECG is continuous like EEG, so this is MSE + the same Laplace rate
term — the codec is literally rian's EEG codec with 63->12 channels.

Reports reconstruction MSE, variance explained, and compression ratio vs float16.

Run:
    python train_ecg.py --epochs 40 --lambda-rate 0.02
"""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

from ecg_data import load_ecg
from ecg_codec import ECGCodec, N_LEADS, N_TIMES


@torch.no_grad()
def evaluate(codec, X, device, bs=256):
    codec.eval()
    mse = bits = n = 0.0
    for i in range(0, X.shape[0], bs):
        x = X[i:i+bs].to(device)
        xh, b = codec.compress_then_reconstruct(x)
        mse += F.mse_loss(xh, x).item() * x.size(0); bits += b.item() * x.size(0); n += x.size(0)
    mse /= n; bps = bits / n
    bpp = codec.bpp_floor(bps)
    return mse, 16.0 / bpp, 1 - mse / float(X.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--c-lat", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-attn", type=int, default=0)
    ap.add_argument("--lambda-rate", type=float, default=0.02)
    ap.add_argument("--out", default="checkpoints/ecg_codec.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    import os; os.makedirs("checkpoints", exist_ok=True)

    train, val, meta = load_ecg()
    tr = torch.from_numpy(train).to(device); va = torch.from_numpy(val).to(device)
    print(f"device={device}  train={meta['n_train']} val={meta['n_val']}  (12 leads × 250)")
    codec = ECGCodec(n_leads=N_LEADS, c_lat=args.c_lat, hidden=args.hidden, n_attn=args.n_attn).to(device)
    print(f"v4 ECGCodec c_lat={args.c_lat} hidden={args.hidden} n_attn={args.n_attn} "
          f"params={sum(p.numel() for p in codec.parameters())}")
    opt = torch.optim.Adam(codec.parameters(), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(0)
    N = tr.shape[0]

    best = 1e9
    for epoch in range(1, args.epochs + 1):
        codec.train(); t0 = time.time(); rm = 0.0; nb = 0
        idx = torch.randperm(N, generator=gen).to(device)
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            x = tr[idx[i:i+args.batch_size]]
            xh, bits, _ = codec(x)
            loss = F.mse_loss(xh, x) + args.lambda_rate * bits
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
            rm += F.mse_loss(xh.detach(), x).item(); nb += 1
        mse, ratio, ve = evaluate(codec, va, device)
        if mse < best:
            best = mse
            torch.save({"model": codec.state_dict(),
                        "config": {"c_lat": args.c_lat, "hidden": args.hidden, "n_attn": args.n_attn, "n_leads": N_LEADS},
                        "meta": {"mean": meta["mean"], "std": meta["std"]},
                        "final_val": {"mse": mse, "ratio": ratio, "var_exp": ve}}, args.out)
        if epoch % max(args.epochs // 12, 1) == 0 or epoch == 1:
            print(f"ep{epoch:2d}/{args.epochs} train_mse={rm/nb:.4f} | val MSE={mse:.4f} "
                  f"var_exp={ve*100:.0f}% ratio={ratio:.0f}x vs fp16 ({time.time()-t0:.1f}s)")
    print(f"\nECG v4 codec: best val MSE={best:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
