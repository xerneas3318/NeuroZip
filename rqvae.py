"""
rqvae.py — RQ-VAE latent upgrade for the NeuroZip codec.

Replaces the scalar-quantize + factorized-Laplace latent with **Residual Vector
Quantization** (the SoundStream / EnCodec approach): the encoder's continuous
latent is quantized by a stack of D codebooks, each refining the residual of the
previous one. The discrete codes (D indices per token) are what you store.

Why it's an upgrade to the latent space:
  - The latent dim (c_lat) can be large/expressive WITHOUT costing bits — the
    rate is fixed by D (num quantizers) x log2(K) (codebook size) per token,
    independent of c_lat. So you decouple representational capacity from bitrate.
  - Codebooks are learned (EMA-updated), giving a data-adaptive discretization
    rather than uniform scalar rounding.

Rate (no entropy model needed): bits/epoch = n_tokens(32) * D * log2(K).
Compression ratio vs float16 = (63*250*16) / bits_per_epoch.

Reuses rian's EEGEncoder / EEGDecoder unchanged.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from codec import EEGEncoder, EEGDecoder, N_CHANNELS, N_TIMES


class VectorQuantizerEMA(nn.Module):
    """Single VQ codebook with EMA updates (stable, avoids codebook-loss tuning)."""

    def __init__(self, dim, codebook_size, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim, self.K, self.decay, self.eps = dim, codebook_size, decay, eps
        embed = torch.randn(codebook_size, dim) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x):                                # x: (B, T, dim)
        flat = x.reshape(-1, self.dim)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1))
        idx = dist.argmin(1)
        q = self.embed[idx].view_as(x)
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.K).type(flat.dtype)
                cluster = onehot.sum(0)
                embed_sum = onehot.t() @ flat
                self.cluster_size.mul_(self.decay).add_(cluster, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
                self.embed.copy_(self.embed_avg / cs.unsqueeze(1))
        q_st = x + (q - x).detach()                     # straight-through
        commit = F.mse_loss(q.detach(), x)              # commitment loss
        return q_st, idx.view(x.shape[:-1]), commit


class ResidualVQ(nn.Module):
    """D codebooks; each quantizes the residual of the previous ones."""

    def __init__(self, dim, codebook_size, num_quantizers):
        super().__init__()
        self.D, self.K = num_quantizers, codebook_size
        self.layers = nn.ModuleList([VectorQuantizerEMA(dim, codebook_size) for _ in range(num_quantizers)])

    def forward(self, x):                               # x: (B, T, dim)
        residual = x
        quantized = torch.zeros_like(x)
        commit = 0.0
        idxs = []
        for layer in self.layers:
            q, idx, c = layer(residual)
            residual = residual - q                     # straight-through makes this residual detached
            quantized = quantized + q
            commit = commit + c
            idxs.append(idx)
        return quantized, torch.stack(idxs, -1), commit / self.D


class EEGCodecRQ(nn.Module):
    """EEG codec with an RQ-VAE latent. Matches EEGCodec's reporting interface."""

    def __init__(self, c_lat=128, hidden=128, n_attn=2, attn_heads=4,
                 codebook_size=1024, num_quantizers=8):
        super().__init__()
        self.encoder = EEGEncoder(c_lat=c_lat, hidden=hidden, n_attn=n_attn, attn_heads=attn_heads)
        self.decoder = EEGDecoder(c_lat=c_lat, hidden=hidden, n_attn=n_attn, attn_heads=attn_heads)
        self.rvq = ResidualVQ(dim=c_lat, codebook_size=codebook_size, num_quantizers=num_quantizers)
        self.c_lat, self.D, self.K, self.n_attn = c_lat, num_quantizers, codebook_size, n_attn
        self.n_tokens = 32

    def forward(self, eeg):
        z = self.encoder(eeg).transpose(1, 2)            # (B, 32, c_lat)
        z_q, idx, commit = self.rvq(z)
        eeg_hat = self.decoder(z_q.transpose(1, 2))
        return eeg_hat, commit, idx                      # idx: (B, 32, D) discrete codes

    @torch.no_grad()
    def compress_then_reconstruct(self, eeg):
        was = self.training; self.eval()
        z = self.encoder(eeg).transpose(1, 2)
        z_q, idx, _ = self.rvq(z)
        eeg_hat = self.decoder(z_q.transpose(1, 2))
        if was: self.train()
        return eeg_hat, torch.tensor(self.bits_per_symbol(), device=eeg.device)

    # ---- reporting (same interface as EEGCodec) ----
    def latent_shape(self):
        return (self.c_lat, self.n_tokens)

    def bits_per_epoch(self):
        return self.n_tokens * self.D * math.log2(self.K)

    def bits_per_symbol(self):
        return self.bits_per_epoch() / (self.c_lat * self.n_tokens)

    def bpp_floor(self, *_):
        return self.bits_per_epoch() / (N_CHANNELS * N_TIMES)

    def compression_ratio(self):
        return 16.0 / self.bpp_floor()
