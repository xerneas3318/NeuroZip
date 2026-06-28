"""
train_v4.py — reproduce rian's v4 fidelity codec, faithfully, to benchmark
RQ-VAE against it.

v4 = conv-only EEGCodec (n_attn=0, c_lat=32, hidden=128), trained on SINGLE
trials with mse + lambda_rate * bits_per_symbol (rian's train.py codec protocol,
15 epochs), evaluated on trial-averaged test MSE. NO averaging / scale-aug
(that's exactly the difference we want to measure).

Run:
    python train_v4.py --lambda-rate 0.004 --epochs 15
"""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from data import ThingsEEG, N_CHANNELS, N_TIMES
from codec import EEGCodec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-rate", type=float, default=0.004)
    ap.add_argument("--c-lat", type=int, default=32)        # v4 defaults
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-attn", type=int, default=0)        # conv-only
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="checkpoints/codec_v4_fidelity.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    tr_ds = ThingsEEG(split="train")                        # single trials, normalized
    tr = torch.from_numpy(tr_ds.eeg).float()                # (66160, 63, 250) on CPU
    te = ThingsEEG(split="test").trial_averaged()[0].to(device)
    print(f"v4 protocol: single-trial train {tuple(tr.shape)} std={tr.std():.3f} "
          f"| trial-avg test {tuple(te.shape)} std={te.std():.3f}")

    codec = EEGCodec(c_lat=args.c_lat, hidden=args.hidden, n_attn=args.n_attn).to(device)
    per_sample = (args.c_lat * 32) / (N_CHANNELS * N_TIMES)
    opt = torch.optim.Adam(codec.parameters(), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(0)
    N = tr.shape[0]
    print(f"v4 EEGCodec c_lat={args.c_lat} hidden={args.hidden} n_attn={args.n_attn} "
          f"params={sum(p.numel() for p in codec.parameters())}")

    @torch.no_grad()
    def ev():
        codec.eval()
        xh, bps = codec.compress_then_reconstruct(te)
        m = F.mse_loss(xh, te).item()
        ratio = 16.0 / codec.bpp_floor(bps.item())
        return m, ratio, 1 - m / te.var().item()

    best = (1e9, 0, 0)
    for epoch in range(1, args.epochs + 1):
        codec.train(); t0 = time.time(); rm = 0.0; nb = 0
        idx = torch.randperm(N, generator=gen)
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            x = tr[idx[i:i+args.batch_size]].to(device, non_blocking=True)
            xh, bps, _ = codec(x)
            loss = F.mse_loss(xh, x) + args.lambda_rate * bps * per_sample
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
            rm += F.mse_loss(xh.detach(), x).item(); nb += 1
        m, ratio, ve = ev()
        if m < best[0]:
            best = (m, ratio, ve)
            torch.save({"model": codec.state_dict(),
                        "config": {"c_lat": args.c_lat, "hidden": args.hidden, "n_attn": args.n_attn},
                        "final_val": {"mse": m, "ratio": ratio, "var_exp": ve}}, args.out)
        print(f"ep{epoch:2d}/{args.epochs} train_mse={rm/nb:.4f} | test-avg MSE={m:.4f} "
              f"ratio={ratio:.0f}x var_exp={ve*100:.0f}% ({time.time()-t0:.1f}s)")
    print(f"\nv4 fidelity best: MSE={best[0]:.4f} ratio={best[1]:.0f}x -> {args.out}")


if __name__ == "__main__":
    main()
