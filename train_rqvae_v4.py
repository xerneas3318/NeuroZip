"""
train_rqvae_v4.py — RQ-VAE trained the SAME WAY as rian's v4, so the comparison
isolates the quantizer (scalar+Laplace vs residual-VQ) and nothing else.

Mirrors train_v4.py exactly:
  - SINGLE-TRIAL training (ThingsEEG.eeg, magnitude ~1.0), no averaging/scale-aug
  - conv-only architecture (c_lat=32, hidden=128, n_attn=0)  [= v4 defaults]
  - 15 epochs, Adam, same lr, grad-clip 1.0
  - evaluated on trial-averaged test MSE (identical eval to train_v4.py)
Only difference: EEGCodecRQ (residual-VQ) instead of EEGCodec (scalar+Laplace).
Rate is matched to v4's ~64x by choosing D,K (D=12,K=1024 -> 32*12*10=3840 bits
-> bpp 0.244 -> 66x), so it's a like-for-like, same-protocol head-to-head.

Run:
    python train_rqvae_v4.py --num-quantizers 12 --codebook-size 1024 --epochs 15
"""
import argparse, time
import torch
import torch.nn.functional as F
from data import ThingsEEG, N_CHANNELS, N_TIMES
from rqvae import EEGCodecRQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-quantizers", type=int, default=12)   # ~66x at K=1024
    ap.add_argument("--codebook-size", type=int, default=1024)
    ap.add_argument("--c-lat", type=int, default=32)            # v4 arch
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-attn", type=int, default=0)            # conv-only, like v4
    ap.add_argument("--epochs", type=int, default=15)           # v4 budget
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--commit", type=float, default=0.25)
    ap.add_argument("--out", default="checkpoints/codec_rqvae_v4proto.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    tr_ds = ThingsEEG(split="train")                            # single trials, normalized
    # Keep the whole single-trial set resident on the GPU (~4 GB) so the GPU
    # isn't starved by per-batch host->device copies (which throttle the CPU).
    tr = torch.from_numpy(tr_ds.eeg).float().to(device)         # (66160, 63, 250)
    te = ThingsEEG(split="test").trial_averaged()[0].to(device)
    print(f"v4 protocol: single-trial train {tuple(tr.shape)} std={tr.std():.3f} "
          f"| trial-avg test {tuple(te.shape)} std={te.std():.3f} var={te.var():.4f}")

    codec = EEGCodecRQ(c_lat=args.c_lat, hidden=args.hidden, n_attn=args.n_attn,
                       codebook_size=args.codebook_size,
                       num_quantizers=args.num_quantizers).to(device)
    ratio = codec.compression_ratio()
    print(f"RQ-VAE (v4 proto): D={args.num_quantizers} K={args.codebook_size} "
          f"c_lat={args.c_lat} n_attn={args.n_attn} -> {codec.bits_per_epoch():.0f} bits/epoch "
          f"ratio={ratio:.0f}x params={sum(p.numel() for p in codec.parameters())}")

    opt = torch.optim.Adam(codec.parameters(), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    N = tr.shape[0]

    @torch.no_grad()
    def ev():
        codec.eval()
        xh, _ = codec.compress_then_reconstruct(te)
        m = F.mse_loss(xh, te).item()
        return m, 1 - m / te.var().item()

    best = (1e9, 0)
    for epoch in range(1, args.epochs + 1):
        codec.train(); t0 = time.time(); rm = 0.0; nb = 0; used = set()
        idx = torch.randperm(N, generator=gen).to(device)
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            x = tr[idx[i:i+args.batch_size]]
            xh, commit, codes = codec(x)
            loss = F.mse_loss(xh, x) + args.commit * commit
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
            rm += F.mse_loss(xh.detach(), x).item(); nb += 1
            used.update(codes[..., 0].unique().tolist())
        m, ve = ev()
        if m < best[0]:
            best = (m, ve)
            torch.save({"model": codec.state_dict(),
                        "config": {"c_lat": args.c_lat, "hidden": args.hidden, "n_attn": args.n_attn,
                                   "codebook_size": args.codebook_size, "num_quantizers": args.num_quantizers},
                        "final_val": {"mse": m, "ratio": ratio, "var_exp": ve}}, args.out)
        print(f"ep{epoch:2d}/{args.epochs} train_mse={rm/nb:.4f} | test-avg MSE={m:.4f} "
              f"var_exp={ve*100:.0f}% ratio={ratio:.0f}x cb0_used={len(used)}/{args.codebook_size} "
              f"({time.time()-t0:.1f}s)")
    print(f"\nRQ-VAE (v4 protocol) best: MSE={best[0]:.4f} var_exp={best[1]*100:.0f}% "
          f"ratio={ratio:.0f}x -> {args.out}")


if __name__ == "__main__":
    main()
