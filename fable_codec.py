"""
fable_codec.py -- a small 3D-conv VQ-VAE for 4D fMRI volumes.

Design goals (from the brief): compress fMRI, keep the model SMALL, use VQ.

  volume (1, 40, 56, 48)
     -> Encoder  (3D conv, n_down stride-2 stages)  -> z_e (D, gd, gh, gw)
     -> VectorQuantizer (EMA codebook, dead-code restart) -> tokens + z_q
     -> Decoder  (mirror, transpose conv)            -> volume_hat

Why VQ and not scalar+Laplace (the original NeuroZip latent)? The RQ-VAE branch
found the two *tie* on fidelity at matched model size -- but VQ gives a genuinely
discrete, fixed-length token grid per volume, which is exactly what lets us
range-code the tokens *across time* (successive fMRI volumes share most tokens).
That temporal coding is where the compression actually comes from here; VQ makes
the bitstream clean to build. We keep the codebook honest with a dead-code
restart (the RQ-VAE branch collapsed to 6/1024 codes without it).

Params stay ~1-2M at the default config -- deliberately small.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class ResBlock3d(nn.Module):
    """Pre-activation residual conv block with LayerScale, keeps spatial size.

    The LayerScale gamma (init 1e-2) is essential here: without it, the
    randomly-initialized residual branches inject enough noise that the whole
    3D AE collapses to predicting the per-voxel mean and never recovers
    (verified empirically -- plain/zero-init/post-norm residual blocks all stall
    at MSE == signal variance). Starting each block near-identity lets gradient
    flow through the skip path and the network actually trains.
    """
    def __init__(self, ch, k=3):
        super().__init__()
        self.net = nn.Sequential(
            _gn(ch), nn.GELU(), nn.Conv3d(ch, ch, k, padding=k // 2),
            _gn(ch), nn.GELU(), nn.Conv3d(ch, ch, k, padding=k // 2))
        self.gamma = nn.Parameter(torch.full((1, ch, 1, 1, 1), 1e-2))

    def forward(self, x):
        return x + self.gamma * self.net(x)


class Encoder(nn.Module):
    """(B,1,D,H,W) -> (B, dim, D/2^n, H/2^n, W/2^n)."""
    def __init__(self, hidden=32, dim=16, n_down=3, depth=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, hidden, 5, padding=2), _gn(hidden), nn.GELU())
        stages, cin = [], hidden
        for i in range(n_down):
            cout = hidden * 2 if i >= 1 else hidden
            stages += [ResBlock3d(cin) for _ in range(depth)]
            stages += [nn.Conv3d(cin, cout, 4, stride=2, padding=1), _gn(cout), nn.GELU()]
            cin = cout
        self.body = nn.Sequential(*stages)
        self.post = nn.Sequential(*[ResBlock3d(cin) for _ in range(depth)])
        self.to_latent = nn.Conv3d(cin, dim, 1)

    def forward(self, x):
        return self.to_latent(self.post(self.body(self.stem(x))))


class Decoder(nn.Module):
    """Mirror of Encoder: (B, dim, gd,gh,gw) -> (B,1,D,H,W)."""
    def __init__(self, hidden=32, dim=16, n_down=3, depth=1):
        super().__init__()
        top = hidden * 2 if n_down >= 1 else hidden
        self.from_latent = nn.Sequential(nn.Conv3d(dim, top, 1), _gn(top), nn.GELU())
        self.pre = nn.Sequential(*[ResBlock3d(top) for _ in range(depth)])
        stages, cin = [], top
        for i in range(n_down):
            cout = hidden if i == n_down - 1 else hidden * 2
            stages += [nn.ConvTranspose3d(cin, cout, 4, stride=2, padding=1), _gn(cout), nn.GELU()]
            stages += [ResBlock3d(cout) for _ in range(depth)]
            cin = cout
        self.body = nn.Sequential(*stages)
        self.head = nn.Conv3d(cin, 1, 3, padding=1)

    def forward(self, z):
        return self.head(self.body(self.pre(self.from_latent(z))))


class VectorQuantizerEMA(nn.Module):
    """EMA-updated codebook with dead-code restart.

    Quantizes the D-dim channel vector at every spatial latent position to one of
    K codebook entries. Returns straight-through z_q (identity gradient to the
    encoder), the integer codes, and the commitment loss.
    """
    def __init__(self, dim, K=512, decay=0.99, eps=1e-5, restart_thresh=1.0):
        super().__init__()
        self.dim, self.K, self.decay, self.eps = dim, K, decay, eps
        self.restart_thresh = restart_thresh
        self.register_buffer("codebook", torch.randn(K, dim) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("ema_sum", self.codebook.clone())

    def _dist(self, flat):
        # ||x - e||^2 = ||x||^2 - 2 x.e + ||e||^2
        return (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.t()
                + self.codebook.pow(2).sum(1))

    def forward(self, z_e):
        # z_e: (B, D, gd, gh, gw)
        B, D = z_e.shape[0], z_e.shape[1]
        grid = z_e.shape[2:]
        flat = z_e.permute(0, 2, 3, 4, 1).reshape(-1, D)          # (N, D)
        idx = self._dist(flat).argmin(1)                          # (N,)
        z_q = self.codebook[idx].view(B, *grid, D).permute(0, 4, 1, 2, 3)

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.K).type_as(flat)     # (N, K)
                n = onehot.sum(0)                                 # counts per code
                self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
                total = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (total + self.K * self.eps) * total
                self.codebook.copy_(self.ema_sum / cs.unsqueeze(1))
                # dead-code restart: revive rarely-used codes from live batch vectors
                dead = self.cluster_size < self.restart_thresh
                if dead.any():
                    n_dead = int(dead.sum())
                    pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
                    self.codebook[dead] = flat[pick]
                    self.ema_sum[dead] = flat[pick]
                    self.cluster_size[dead] = 1.0

        commit = F.mse_loss(z_e, z_q.detach())
        z_q_st = z_e + (z_q - z_e).detach()                       # straight-through
        return z_q_st, idx.view(B, *grid), commit

    def decode_codes(self, codes):
        """codes: (B, gd, gh, gw) int -> z_q (B, D, gd, gh, gw)."""
        B, grid = codes.shape[0], codes.shape[1:]
        z_q = self.codebook[codes.reshape(-1)].view(B, *grid, self.dim)
        return z_q.permute(0, 4, 1, 2, 3)


class VQCodec(nn.Module):
    def __init__(self, vol_shape=(40, 56, 48), hidden=32, dim=16,
                 n_down=3, depth=1, K=512, commit_beta=0.25):
        super().__init__()
        self.vol_shape = tuple(vol_shape)
        self.encoder = Encoder(hidden, dim, n_down, depth)
        self.decoder = Decoder(hidden, dim, n_down, depth)
        self.vq = VectorQuantizerEMA(dim, K=K)
        self.commit_beta = commit_beta
        self.K, self.dim, self.n_down = K, dim, n_down
        self.grid = tuple(s // (2 ** n_down) for s in self.vol_shape)
        self.tokens_per_vol = int(self.grid[0] * self.grid[1] * self.grid[2])

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, codes, commit = self.vq(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, codes, self.commit_beta * commit

    @torch.no_grad()
    def encode(self, x):
        return self.vq(self.encoder(x))[1]                        # integer codes

    @torch.no_grad()
    def decode(self, codes):
        return self.decoder(self.vq.decode_codes(codes))

    def raw_bits_per_vol(self):
        """VQ token rate before temporal entropy coding."""
        import math
        return self.tokens_per_vol * math.log2(self.K)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    import math
    m = VQCodec()
    x = torch.randn(2, 1, 40, 56, 48)
    xh, codes, closs = m(x)
    print(f"params           : {count_params(m)/1e6:.2f}M")
    print(f"input            : {tuple(x.shape)}")
    print(f"latent grid      : {m.grid}  -> {m.tokens_per_vol} tokens/vol, K={m.K}")
    print(f"recon            : {tuple(xh.shape)}   codes {tuple(codes.shape)}")
    print(f"raw bits/vol     : {m.raw_bits_per_vol():.0f}  ({m.raw_bits_per_vol()/8:.0f} bytes, pre temporal coding)")
    assert xh.shape == x.shape
    dec = m.decode(codes)
    print(f"decode(codes)    : {tuple(dec.shape)}  OK")
