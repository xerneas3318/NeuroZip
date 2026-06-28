"""
fmri_codec.py — the v4 NeuroZip codec, adapted to fMRI.

The raw fMRI volume is 3-D+time (huge, hard). The right 1-D representation is the
**parcellated ROI time series**: (regions x time). BOLD is slow/smooth, so this
(channels x length) signal compresses well with the same conv codec. Like the EEG
and ECG branches, this is rian's EEG codec with one change: 63 EEG channels ->
200 brain ROIs (CC200 atlas). Continuous signal -> MSE, same Laplace rate model.

Reuses rian's exact blocks (ConvNormAct, FactorizedLaplacePrior, quantize).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec import ConvNormAct, TransformerStack, FactorizedLaplacePrior, quantize

N_ROI = 200        # CC200 atlas regions  (was N_CHANNELS = 63 for EEG)
N_TIMES = 64       # TRs per window (64 = 8*8, downsamples to 8)


class fMRIEncoder(nn.Module):
    """(B, 200, 128) -> (B, c_lat, 8). Same as EEGEncoder, in_ch=200, len 128."""

    def __init__(self, n_roi=N_ROI, c_lat=32, hidden=128, n_attn=0, attn_heads=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_roi, hidden, kernel_size=15, padding=7),
            nn.GroupNorm(8, hidden), nn.GELU())
        self.body = nn.Sequential(                              # 64 -> 32 -> 16 -> 8
            ConvNormAct(hidden, hidden, k=5, stride=2),
            ConvNormAct(hidden, hidden * 2, k=5, stride=2),
            ConvNormAct(hidden * 2, hidden * 2, k=3, stride=2))
        self.n_attn = n_attn
        if n_attn > 0:
            self.attn = TransformerStack(dim=hidden * 2, n_layers=n_attn, max_len=32, n_heads=attn_heads)
        self.to_latent = nn.Conv1d(hidden * 2, c_lat, kernel_size=1)
        self.c_lat = c_lat

    def forward(self, x):
        x = self.body(self.stem(x))                            # 128 -> 16 (no pad needed)
        if self.n_attn > 0:
            x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        return self.to_latent(x)


class fMRIDecoder(nn.Module):
    """(B, c_lat, 8) -> (B, 200, 64). Mirror of fMRIEncoder."""

    def __init__(self, n_roi=N_ROI, c_lat=32, hidden=128, n_attn=0, attn_heads=4):
        super().__init__()
        self.from_latent = nn.Conv1d(c_lat, hidden * 2, kernel_size=1)
        self.n_attn = n_attn
        if n_attn > 0:
            self.attn = TransformerStack(dim=hidden * 2, n_layers=n_attn, max_len=32, n_heads=attn_heads)
        self.body = nn.Sequential(                              # 8 -> 16 -> 32 -> 64
            ConvNormAct(hidden * 2, hidden * 2, k=4, stride=2, transpose=True),
            ConvNormAct(hidden * 2, hidden, k=4, stride=2, transpose=True),
            ConvNormAct(hidden, hidden, k=4, stride=2, transpose=True))
        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(hidden, n_roi, kernel_size=1))

    def forward(self, z):
        x = self.from_latent(z)
        if self.n_attn > 0:
            x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(self.body(x))                         # already 128


class fMRICodec(nn.Module):
    """Same structure as EEGCodec. n_attn=0 = conv-only v4 codec."""

    def __init__(self, n_roi=N_ROI, c_lat=32, hidden=128, n_attn=0, attn_heads=4):
        super().__init__()
        self.encoder = fMRIEncoder(n_roi, c_lat, hidden, n_attn, attn_heads)
        self.decoder = fMRIDecoder(n_roi, c_lat, hidden, n_attn, attn_heads)
        self.prior = FactorizedLaplacePrior(c_lat=c_lat)
        self.c_lat, self.n_roi, self.n_times, self.n_attn = c_lat, n_roi, N_TIMES, n_attn

    def forward(self, x):
        y = self.encoder(x)
        y_q = quantize(y, self.training)
        bits_per_symbol = self.prior.bits(y_q)
        x_hat = self.decoder(y_q)
        return x_hat, bits_per_symbol, y_q

    @torch.no_grad()
    def compress_then_reconstruct(self, x):
        was = self.training; self.eval()
        y = self.encoder(x); y_int = torch.round(y)
        bits = self.prior.bits(y_int); x_hat = self.decoder(y_int)
        if was: self.train()
        return x_hat, bits

    def latent_shape(self):
        return (self.c_lat, 8)

    def bpp_floor(self, bits_per_symbol):
        return bits_per_symbol * self.c_lat * 8 / (self.n_roi * self.n_times)
