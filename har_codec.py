"""
har_codec.py -- the 1-D NeuroZip codec on (9 channels x 128) HAR windows.
Same family as the EEG/ECG codec (rian's blocks), sized for a short 9-channel
window. 128 = 2^7 so three stride-2 stages land on 16 latent steps, no padding.
"""
import torch
import torch.nn as nn
from codec import ConvNormAct, FactorizedLaplacePrior, quantize

N_CH, N_TIMES = 9, 128


class HAREncoder(nn.Module):
    def __init__(self, n_ch=N_CH, c_lat=16, hidden=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_ch, hidden, 7, padding=3),
                                  nn.GroupNorm(8, hidden), nn.GELU())
        self.body = nn.Sequential(
            ConvNormAct(hidden, hidden, k=5, stride=2),        # 64
            ConvNormAct(hidden, hidden * 2, k=5, stride=2),    # 32
            ConvNormAct(hidden * 2, hidden * 2, k=3, stride=2))  # 16
        self.to_latent = nn.Conv1d(hidden * 2, c_lat, 1)

    def forward(self, x):
        return self.to_latent(self.body(self.stem(x)))


class HARDecoder(nn.Module):
    def __init__(self, n_ch=N_CH, c_lat=16, hidden=64):
        super().__init__()
        self.from_latent = nn.Conv1d(c_lat, hidden * 2, 1)
        self.body = nn.Sequential(
            ConvNormAct(hidden * 2, hidden * 2, k=4, stride=2, transpose=True),  # 32
            ConvNormAct(hidden * 2, hidden, k=4, stride=2, transpose=True),      # 64
            ConvNormAct(hidden, hidden, k=4, stride=2, transpose=True))          # 128
        self.head = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(hidden, n_ch, 1))

    def forward(self, z):
        return self.head(self.body(self.from_latent(z)))


class HARCodec(nn.Module):
    def __init__(self, n_ch=N_CH, c_lat=16, hidden=64):
        super().__init__()
        self.encoder = HAREncoder(n_ch, c_lat, hidden)
        self.decoder = HARDecoder(n_ch, c_lat, hidden)
        self.prior = FactorizedLaplacePrior(c_lat=c_lat)
        self.c_lat, self.n_ch = c_lat, n_ch
        self.latent_t = N_TIMES // 8

    def forward(self, x):
        y = self.encoder(x); yq = quantize(y, self.training)
        return self.decoder(yq), self.prior.bits(yq), yq

    @torch.no_grad()
    def compress_then_reconstruct(self, x):
        was = self.training; self.eval()
        y = torch.round(self.encoder(x)); out = self.decoder(y), self.prior.bits(y)
        if was: self.train()
        return out

    def bpp_floor(self, bps):
        return bps * self.c_lat * self.latent_t / (self.n_ch * N_TIMES)
