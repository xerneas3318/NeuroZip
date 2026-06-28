"""NeuroZip RQ-VAE codec (residual vector-quantized variant).

A second compression model that drops in alongside the continuous EEGCodec. It reuses
the same conv encoder/decoder, but replaces the scalar round()+Laplace-prior bottleneck
with a **residual vector quantizer**: n_q stages, each a learned codebook of K vectors,
quantizing the per-timestep latent vector. The bitrate is fixed by (n_q, K):
    bits = n_q * T_lat * log2(K)
so different (n_q, K) give the tier ladder, and the model is trained to make those
discrete codes both reconstruct well and stay retrievable (task-aware, same frozen judge).

Interface mirrors EEGCodec so it slots into serve.py / evaluate viewers:
    .encoder, .compress_then_reconstruct(eeg) -> (eeg_hat, bits), .bpp_floor()
forward(eeg) -> (eeg_hat, vq_loss, bits_per_symbol, codes)
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import EEGEncoder, EEGDecoder, N_CHANNELS, N_TIMES, CHECKPOINTS

T_LAT = 32  # latent timesteps (encoder downsamples 250 -> 32)


class ResidualVQ(nn.Module):
    """Multi-stage residual vector quantizer over per-timestep latent vectors."""

    def __init__(self, dim: int, n_q: int = 4, codebook_size: int = 256, beta: float = 0.25):
        super().__init__()
        self.dim, self.n_q, self.K, self.beta = dim, n_q, codebook_size, beta
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, dim) for _ in range(n_q)])
        for cb in self.codebooks:
            cb.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor):
        # z: (B, dim, T) -> quantize each (dim,) vector at each timestep
        B, D, T = z.shape
        x = z.permute(0, 2, 1).reshape(B * T, D)         # (B*T, dim)
        residual = x
        total_q = torch.zeros_like(x)
        vq_loss = x.new_zeros(())
        codes = []
        for cb in self.codebooks:
            e = cb.weight                                # (K, dim)
            d = (residual.pow(2).sum(1, keepdim=True)
                 - 2 * residual @ e.t()
                 + e.pow(2).sum(1))                      # (B*T, K)
            idx = d.argmin(1)
            q = cb(idx)                                  # (B*T, dim)
            # codebook loss (move codes to encoder output) + commitment loss
            vq_loss = vq_loss + F.mse_loss(q, residual.detach()) \
                + self.beta * F.mse_loss(residual, q.detach())
            total_q = total_q + q
            residual = residual - q.detach()
            codes.append(idx)
        # straight-through: gradient of total_q flows to encoder as identity
        z_q = x + (total_q - x).detach()
        z_q = z_q.reshape(B, T, D).permute(0, 2, 1)
        codes = torch.stack(codes, dim=1).reshape(B, T, self.n_q)
        return z_q, vq_loss, codes


class EEGRQVAE(nn.Module):
    def __init__(self, c_lat: int = 32, hidden: int = 128, n_attn: int = 0,
                 n_q: int = 4, codebook_size: int = 256, beta: float = 0.25):
        super().__init__()
        self.encoder = EEGEncoder(c_lat=c_lat, hidden=hidden, n_attn=n_attn)
        self.decoder = EEGDecoder(c_lat=c_lat, hidden=hidden, n_attn=n_attn)
        self.rvq = ResidualVQ(c_lat, n_q=n_q, codebook_size=codebook_size, beta=beta)
        self.c_lat, self.hidden, self.n_attn = c_lat, hidden, n_attn
        self.n_q, self.K = n_q, codebook_size

    def _bits_per_symbol(self) -> float:
        # Fixed-rate code; expressed per latent scalar so bpp_floor() matches EEGCodec.
        total_bits = self.n_q * T_LAT * math.log2(self.K)
        return total_bits / (self.c_lat * T_LAT)

    def bpp_floor(self, bits_per_symbol: float | None = None) -> float:
        total_bits = self.n_q * T_LAT * math.log2(self.K)
        return total_bits / (N_CHANNELS * N_TIMES)

    def forward(self, eeg: torch.Tensor):
        z = self.encoder(eeg)
        z_q, vq_loss, codes = self.rvq(z)
        eeg_hat = self.decoder(z_q)
        return eeg_hat, vq_loss, self._bits_per_symbol(), codes

    @torch.no_grad()
    def compress_then_reconstruct(self, eeg: torch.Tensor):
        z = self.encoder(eeg)
        z_q, _, _ = self.rvq(z)
        eeg_hat = self.decoder(z_q)
        return eeg_hat, torch.tensor(self._bits_per_symbol(), device=eeg.device)


def compression_ratio(model: EEGRQVAE, bits_per_symbol: float = None) -> float:
    return 16.0 / max(model.bpp_floor(), 1e-6)
