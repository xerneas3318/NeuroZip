"""
train_fidelity.py — fidelity benchmark: rian's scalar EEGCodec (round + Laplace),
trained with the same protocol as the RQ-VAE (averaged train + scale-aug) so the
viewer can show it as a matched-compression baseline.

Run:
    python train_fidelity.py --lambda-rate 0.01 --c-lat 96 --hidden 160 --epochs 100
"""
import argparse, time
import torch
import torch.nn.functional as F
from data import ThingsEEG, N_CHANNELS, N_TIMES
from codec import EEGCodec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-rate", type=float, default=0.01)
    ap.add_argument("--c-lat", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=160)
    ap.add_argument("--n-attn", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--scale-aug", type=float, default=0.25)
    ap.add_argument("--out", default="checkpoints/codec_fidelity_72x.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    tr = ThingsEEG(split="train").trial_averaged()[0].to(device)
    te = ThingsEEG(split="test").trial_averaged()[0].to(device)
    codec = EEGCodec(c_lat=args.c_lat, hidden=args.hidden, n_attn=args.n_attn).to(device)
    nsym = args.c_lat * 32
    per_sample = nsym / (N_CHANNELS * N_TIMES)
    print(f"fidelity EEGCodec c_lat={args.c_lat} hidden={args.hidden} -> {nsym} symbols/epoch  "
          f"params={sum(p.numel() for p in codec.parameters())}")

    opt = torch.optim.Adam(codec.parameters(), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(0)
    N = tr.shape[0]

    @torch.no_grad()
    def ev():
        codec.eval()
        xh, bps = codec.compress_then_reconstruct(te)
        m = F.mse_loss(xh, te).item()
        ratio = 16.0 / codec.bpp_floor(bps.item())
        return m, ratio, codec.bpp_floor(bps.item()), 1 - m / te.var().item()

    best = 1e9
    for epoch in range(1, args.epochs + 1):
        codec.train(); t0 = time.time(); rm = 0.0; nb = 0
        idx = torch.randperm(N, generator=gen).to(device)
        for i in range(0, N - args.batch_size + 1, args.batch_size):
            x = tr[idx[i:i+args.batch_size]]
            s = args.scale_aug + (1 - args.scale_aug) * torch.rand(x.size(0), 1, 1, device=device)
            x = x * s
            xh, bps, _ = codec(x)
            loss = F.mse_loss(xh, x) + args.lambda_rate * bps * per_sample
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0); opt.step()
            rm += F.mse_loss(xh.detach(), x).item(); nb += 1
        m, ratio, bpp, ve = ev()
        if m < best:
            best = m
            torch.save({"model": codec.state_dict(),
                        "config": {"c_lat": args.c_lat, "hidden": args.hidden, "n_attn": args.n_attn},
                        "final_val": {"mse": m, "ratio": ratio, "bpp": bpp, "var_exp": ve}}, args.out)
        if epoch % max(args.epochs // 12, 1) == 0 or epoch == 1:
            print(f"ep{epoch:3d}/{args.epochs} train_mse={rm/nb:.4f} | test-avg MSE={m:.4f} "
                  f"bpp={bpp:.3f} ratio={ratio:.0f}x var_exp={ve*100:.0f}% ({time.time()-t0:.1f}s)")
    print(f"\nfidelity benchmark best MSE={best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
