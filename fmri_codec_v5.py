"""
fmri_codec_v5.py — a HEAVIER codec at the SAME latent budget as v4.

Goal: keep compression constant (same number of latent symbols as the original
v4 fMRI codec, 1024 = c_lat 128 x 8) but drive MSE down by improving the
*architecture* instead of the bitrate:
  - residual conv blocks (depth) at every resolution
  - transformer attention at the bottleneck (cross-time mixing)
  - a smarter latent split: c_lat 64 x 16 (keeps 2x more temporal than 128 x 8)
  - wider hidden

Default: input (200, 64) -> latent (c_lat, 16) -> (200, 64). n_down=2 (4x time).
Reuses rian's FactorizedLaplacePrior + quantize + TransformerStack.
"""
import torch
import torch.nn as nn

from codec import FactorizedLaplacePrior, TransformerStack, quantize

N_ROI = 200
N_TIMES = 64


class ResBlock(nn.Module):
    """Pre-activation residual conv block (keeps length)."""
    def __init__(self, ch, k=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.GELU(), nn.Conv1d(ch, ch, k, padding=k // 2),
            nn.GroupNorm(8, ch), nn.GELU(), nn.Conv1d(ch, ch, k, padding=k // 2))

    def forward(self, x):
        return x + self.net(x)


def _down(cin, cout):
    return nn.Conv1d(cin, cout, kernel_size=4, stride=2, padding=1)      # halve length


def _up(cin, cout):
    return nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1)  # double length


class HeavyEncoder(nn.Module):
    """(B, 200, 64) -> (B, c_lat, 16).  n_down stride-2 stages, depth res blocks each."""
    def __init__(self, n_roi=N_ROI, c_lat=64, hidden=192, depth=3, n_attn=4, n_down=2):
        super().__init__()
        H = hidden
        self.stem = nn.Sequential(nn.Conv1d(n_roi, H, 15, padding=7), nn.GroupNorm(8, H), nn.GELU())
        stages, cin = [], H
        for i in range(n_down):
            cout = H * 2 if i == n_down - 1 else H
            blk = [ResBlock(cin) for _ in range(depth)]
            blk.append(_down(cin, cout)); blk.append(nn.GELU())
            stages += blk; cin = cout
        self.body = nn.Sequential(*stages)
        self.post = nn.Sequential(*[ResBlock(cin) for _ in range(depth)])
        self.attn = TransformerStack(dim=cin, n_layers=n_attn, max_len=64, n_heads=8) if n_attn else None
        self.to_latent = nn.Conv1d(cin, c_lat, 1)
        self.c_lat = c_lat

    def forward(self, x):
        x = self.post(self.body(self.stem(x)))
        if self.attn is not None:
            x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        return self.to_latent(x)


class HeavyDecoder(nn.Module):
    """(B, c_lat, 16) -> (B, 200, 64). Mirror of HeavyEncoder."""
    def __init__(self, n_roi=N_ROI, c_lat=64, hidden=192, depth=3, n_attn=4, n_down=2):
        super().__init__()
        H = hidden
        top = H * 2
        self.from_latent = nn.Conv1d(c_lat, top, 1)
        self.attn = TransformerStack(dim=top, n_layers=n_attn, max_len=64, n_heads=8) if n_attn else None
        self.pre = nn.Sequential(*[ResBlock(top) for _ in range(depth)])
        stages, cin = [], top
        for i in range(n_down):
            cout = H if i >= n_down - 1 else top
            stages.append(_up(cin, cout)); stages.append(nn.GELU())
            stages += [ResBlock(cout) for _ in range(depth)]; cin = cout
        self.body = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.Conv1d(cin, H, 7, padding=3), nn.GELU(), nn.Conv1d(H, n_roi, 1))

    def forward(self, z):
        x = self.from_latent(z)
        if self.attn is not None:
            x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(self.body(self.pre(x)))


class HeavyfMRICodec(nn.Module):
    def __init__(self, n_roi=N_ROI, c_lat=64, hidden=192, depth=3, n_attn=4, n_down=2, **kw):
        super().__init__()
        self.encoder = HeavyEncoder(n_roi, c_lat, hidden, depth, n_attn, n_down)
        self.decoder = HeavyDecoder(n_roi, c_lat, hidden, depth, n_attn, n_down)
        self.prior = FactorizedLaplacePrior(c_lat=c_lat)
        self.c_lat, self.n_roi, self.n_times = c_lat, n_roi, N_TIMES
        self.latent_t = N_TIMES // (2 ** n_down)

    def forward(self, x):
        y = self.encoder(x); y_q = quantize(y, self.training)
        return self.decoder(y_q), self.prior.bits(y_q), y_q

    @torch.no_grad()
    def compress_then_reconstruct(self, x):
        was = self.training; self.eval()
        y = torch.round(self.encoder(x)); out = self.decoder(y), self.prior.bits(y)
        if was: self.train()
        return out

    def bpp_floor(self, bits_per_symbol):
        return bits_per_symbol * self.c_lat * self.latent_t / (self.n_roi * self.n_times)
