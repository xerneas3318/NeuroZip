"""
train_rqvae.py — train the RQ-VAE codec and report trial-averaged test MSE.

Same protocol as the winning scalar codec: train on the full averaged train
split (16,540 images), evaluate MSE on the 200 trial-averaged test concepts
(single-trial-std normalization = the README's scale), with scale augmentation
to bridge the 4-rep-train vs 80-rep-test magnitude gap.

Rate is set by the number of residual quantizers D (and codebook size K), so
--num-quantizers picks the compression tier (D=11,K=1024 ~ 72x).

Run:
    python train_rqvae.py --num-quantizers 11 --codebook-size 1024 --epochs 100
"""

import argparse
import time
import torch
import torch.nn.functional as F

from data import ThingsEEG, N_CHANNELS, N_TIMES
from rqvae import EEGCodecRQ

CKPTS = __import__("pathlib").Path("checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-quantizers", type=int, default=11)
    ap.add_argument("--codebook-size", type=int, default=1024)
    ap.add_argument("--c-lat", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--n-attn", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--commit", type=float, default=0.25)
    ap.add_argument("--scale-aug", type=float, default=0.25)
    ap.add_argument("--out", default="checkpoints/codec_rqvae.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    tr = ThingsEEG(split="train").trial_averaged()[0].to(device)   # (16540, 63, 250) avg, single-trial norm
    te = ThingsEEG(split="test").trial_averaged()[0].to(device)    # (200, 63, 250)
    print(f"train {tuple(tr.shape)} std={tr.std():.3f} | test {tuple(te.shape)} std={te.std():.3f} var={te.var():.4f}")

    codec = EEGCodecRQ(c_lat=args.c_lat, hidden=args.hidden, n_attn=args.n_attn,
                       codebook_size=args.codebook_size, num_quantizers=args.num_quantizers).to(device)
    ratio = codec.compression_ratio()
    print(f"RQ-VAE: D={args.num_quantizers} K={args.codebook_size} c_lat={args.c_lat} -> "
          f"{codec.bits_per_epoch():.0f} bits/epoch  ratio={ratio:.0f}x  params={sum(p.numel() for p in codec.parameters())}")

    opt = torch.optim.Adam(codec.parameters(), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(0)
    N = tr.shape[0]

    @torch.no_grad()
    def eval_mse():
        codec.eval()
        xh, _ = codec.compress_then_reconstruct(te)
        m = F.mse_loss(xh, te).item()
        return m, 1 - m / te.var().item()

    best = 1e9
    for epoch in range(1, args.epochs + 1):
        codec.train(); t0 = time.time(); rm = 0.0; nb = 0; used = set()
        idx = torch.randperm(N, generator=gen).to(device)
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            x = tr[idx[i:i+args.batch_size]]
            s = args.scale_aug + (1 - args.scale_aug) * torch.rand(x.size(0), 1, 1, device=device)
            x = x * s
            xh, commit, codes = codec(x)
            loss = F.mse_loss(xh, x) + args.commit * commit
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
            rm += F.mse_loss(xh.detach(), x).item(); nb += 1
            used.update(codes[..., 0].unique().tolist())
        m, ve = eval_mse()
        if m < best:
            best = m
            torch.save({"model": codec.state_dict(),
                        "config": {"c_lat": args.c_lat, "hidden": args.hidden, "n_attn": args.n_attn,
                                   "codebook_size": args.codebook_size, "num_quantizers": args.num_quantizers},
                        "final_val": {"mse": m, "ratio": ratio, "var_exp": ve}}, args.out)
        if epoch % max(args.epochs // 12, 1) == 0 or epoch == 1:
            print(f"ep{epoch:3d}/{args.epochs} train_mse={rm/nb:.4f} | test-avg MSE={m:.4f} "
                  f"var_exp={ve*100:.0f}% cb0_used={len(used)}/{args.codebook_size} ({time.time()-t0:.1f}s)")
    print(f"\nRQ-VAE best test-avg MSE={best:.4f} at {ratio:.0f}x  (README neurozip_high 0.0246 @ 74x) -> {args.out}")


if __name__ == "__main__":
    main()
