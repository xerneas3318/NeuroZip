"""
protein_codec.py — the v4 NeuroZip codec, adapted to PROTEINS.

Demonstrates the architecture is domain-agnostic: this is byte-for-byte the same
conv encoder/decoder + factorized-Laplace prior + noise/round quantizer as rian's
EEG codec (codec.py). The ONLY change is the channel count: 63 EEG channels ->
20 amino acids. A protein of length L is represented as a (20 x L) one-hot
"image" (20 channels = amino acids, L positions = residues), exactly analogous to
the (63 channels x time) EEG epoch. Same downsampling 250->32, same Laplace rate
model, same scalar quantizer. No VQ / RQ — pure v4.

Reuses rian's building blocks directly (ConvNormAct, TransformerStack,
FactorizedLaplacePrior, quantize) so the equivalence is obvious.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec import ConvNormAct, TransformerStack, FactorizedLaplacePrior, quantize

N_AA = 20          # 20 standard amino acids  (was N_CHANNELS = 63 for EEG)
N_POS = 250        # residue positions        (was N_TIMES = 250 for EEG)
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


class ProteinEncoder(nn.Module):
    """(B, 20, L) -> (B, c_lat, 32). Identical to EEGEncoder, in_ch = 20."""

    def __init__(self, n_aa=N_AA, c_lat=32, hidden=128, n_attn=0, attn_heads=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_aa, hidden, kernel_size=15, padding=7),
            nn.GroupNorm(8, hidden), nn.GELU())
        self.body = nn.Sequential(                              # 256 -> 128 -> 64 -> 32
            ConvNormAct(hidden, hidden, k=5, stride=2),
            ConvNormAct(hidden, hidden * 2, k=5, stride=2),
            ConvNormAct(hidden * 2, hidden * 2, k=3, stride=2))
        self.n_attn = n_attn
        if n_attn > 0:
            self.attn = TransformerStack(dim=hidden * 2, n_layers=n_attn, max_len=64, n_heads=attn_heads)
        self.to_latent = nn.Conv1d(hidden * 2, c_lat, kernel_size=1)
        self.c_lat = c_lat

    def forward(self, x):
        x = F.pad(x, (3, 3), mode="reflect")                   # 250 -> 256
        x = self.body(self.stem(x))
        if self.n_attn > 0:
            x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        return self.to_latent(x)


class ProteinDecoder(nn.Module):
    """(B, c_lat, 32) -> (B, 20, L). Identical to EEGDecoder, out_ch = 20."""

    def __init__(self, n_aa=N_AA, c_lat=32, hidden=128, n_attn=0, attn_heads=4):
        super().__init__()
        self.from_latent = nn.Conv1d(c_lat, hidden * 2, kernel_size=1)
        self.n_attn = n_attn
        if n_attn > 0:
            self.attn = TransformerStack(dim=hidden * 2, n_layers=n_attn, max_len=64, n_heads=attn_heads)
        self.body = nn.Sequential(                              # 32 -> 64 -> 128 -> 256
            ConvNormAct(hidden * 2, hidden * 2, k=4, stride=2, transpose=True),
            ConvNormAct(hidden * 2, hidden, k=4, stride=2, transpose=True),
            ConvNormAct(hidden, hidden, k=4, stride=2, transpose=True))
        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3), nn.GELU(),
            nn.Conv1d(hidden, n_aa, kernel_size=1))

    def forward(self, z):
        x = self.from_latent(z)
        if self.n_attn > 0:
            x = self.attn(x.transpose(1, 2)).transpose(1, 2)
        x = self.head(self.body(x))
        return x[..., 3:-3]                                     # 256 -> 250


class ProteinCodec(nn.Module):
    """Same structure as EEGCodec. n_attn=0 = the conv-only v4 codec."""

    def __init__(self, n_aa=N_AA, c_lat=32, hidden=128, n_attn=0, attn_heads=4):
        super().__init__()
        self.encoder = ProteinEncoder(n_aa, c_lat, hidden, n_attn, attn_heads)
        self.decoder = ProteinDecoder(n_aa, c_lat, hidden, n_attn, attn_heads)
        self.prior = FactorizedLaplacePrior(c_lat=c_lat)
        self.c_lat, self.n_aa, self.n_pos, self.n_attn = c_lat, n_aa, N_POS, n_attn

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
        """bits per (one-hot) input sample = bits/symbol * #symbols / #samples."""
        return bits_per_symbol * self.c_lat * 32 / (self.n_aa * self.n_pos)
