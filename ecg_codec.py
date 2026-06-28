"""
ecg_codec.py — the v4 NeuroZip codec, adapted to ECG.

ECG is the cleanest "good for reconstruction" target: a continuous (leads × time)
signal, highly redundant (the heartbeat repeats), so the same conv codec that
compresses EEG ~70× compresses ECG hard too. Like the protein branch, this is
byte-for-byte rian's EEG codec with one change: 63 EEG channels -> 12 ECG leads.
ECG is continuous (like EEG), so the loss is MSE — no categorical adaptation.

Reuses rian's exact blocks (ConvNormAct, FactorizedLaplacePrior, quantize).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec import ConvNormAct, TransformerStack, FactorizedLaplacePrior, quantize

N_LEADS = 12       # 12-lead ECG  (was N_CHANNELS = 63 for EEG)
N_TIMES = 250      # samples per epoch (2.5 s @ 100 Hz)  (was 250 for EEG)
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGEncoder(nn.Module):
    """(B, 12, 250) -> (B, c_lat, 32). Identical to EEGEncoder, in_ch = 12."""

    def __init__(self, n_leads=N_LEADS, c_lat=32, hidden=128, n_attn=0, attn_heads=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, hidden, kernel_size=15, padding=7),
            nn.GroupNorm(8, hidden), nn.GELU())
        self.body = nn.Sequential(
            ConvNormAct(hidden, hidden, k=5, stride=2),
            ConvNormAct(hidden, hidden * 2, k=5, stride=2),
            ConvNormAct(hidden * 2, hidden * 2, k=3, stride=2))
        self.n_attn = n_attn
        if n_attn > 0:
            self.attn = TransformerStack(dim=hidden * 2, n_layers=n_attn, max_len=64, n_heads=attn_heads)
        self.to_latent = nn.Conv1d(hidden * 2, c_lat, kernel_size=1)
        self.c_lat = c_lat

    def forward(self, x):
        x = F.pad(x, (3, 3), mode="reflect")           # 250 -> 256
        x = self.body(self.stem(x))
        if self.n_attn > 0:
            x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        return self.to_latent(x)


class ECGDecoder(nn.Module):
    """(B, c_lat, 32) -> (B, 12, 250). Identical to EEGDecoder, out_ch = 12."""

    def __init__(self, n_leads=N_LEADS, c_lat=32, hidden=128, n_attn=0, attn_heads=4):
        super().__init__()
        self.from_latent = nn.Conv1d(c_lat, hidden * 2, kernel_size=1)
        self.n_attn = n_attn
        if n_attn > 0:
            self.attn = TransformerStack(dim=hidden * 2, n_layers=n_attn, max_len=64, n_heads=attn_heads)
        self.body = nn.Sequential(
            ConvNormAct(hidden * 2, hidden * 2, k=4, stride=2, transpose=True),
            ConvNormAct(hidden * 2, hidden, k=4, stride=2, transpose=True),
            ConvNormAct(hidden, hidden, k=4, stride=2, transpose=True))
        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(hidden, n_leads, kernel_size=1))

    def forward(self, z):
        x = self.from_latent(z)
        if self.n_attn > 0:
            x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        x = self.head(self.body(x))
        return x[..., 3:-3]                             # 256 -> 250


class ECGCodec(nn.Module):
    """Same structure as EEGCodec. n_attn=0 = conv-only v4 codec."""

    def __init__(self, n_leads=N_LEADS, c_lat=32, hidden=128, n_attn=0, attn_heads=4):
        super().__init__()
        self.encoder = ECGEncoder(n_leads, c_lat, hidden, n_attn, attn_heads)
        self.decoder = ECGDecoder(n_leads, c_lat, hidden, n_attn, attn_heads)
        self.prior = FactorizedLaplacePrior(c_lat=c_lat)
        self.c_lat, self.n_leads, self.n_times, self.n_attn = c_lat, n_leads, N_TIMES, n_attn

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
        return (self.c_lat, 32)

    def bpp_floor(self, bits_per_symbol):
        return bits_per_symbol * self.c_lat * 32 / (self.n_leads * self.n_times)
