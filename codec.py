"""NeuroZip EEG codec (Stage 2).

A 1D-conv autoencoder over time, with per-channel awareness preserved. The
encoder downsamples (63, 250) -> (C_lat, T_lat), each scalar is quantized to
integers, and the decoder upsamples back to (63, 250).

THREE EASY-TO-GET-WRONG SPOTS (also reiterated in train.py & evaluate.py):

  (1) Quantizer:
      Training:   y = y + Uniform(-0.5, 0.5)   (differentiable proxy)
      Inference:  y = round(y)                 (real integer codes)
      Mismatch on this is the most common bug: forget the noise at train
      time and the decoder learns a continuous latent that round() breaks.

  (2) Rate estimate:
      We use a factorized prior with per-channel Laplace(0, b_c) where b_c is
      a learnable scale. bits = -log2 prob(round(y) in [y-0.5, y+0.5]). At
      training time we sub the noised latent for round(y) so it's differen-
      tiable. This number is the *actual* compressed size if you arithmetic-
      coded with this prior, modulo bookkeeping; we report it as bpp.

  (3) Normalization stats (in data.py):
      The codec sees normalized EEG. To report bits-per-sample-in-real-units
      you must remember the per-channel mean/std and apply them in reverse
      for fidelity comparisons. We save them once and reuse.

For task-aware training (Stage 3), the loss adds a CLIP-space term routed
through the frozen projector P. The projector and CLIP features are FROZEN
but gradient flow is NOT blocked through them.
"""

from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINTS = Path(__file__).parent / "checkpoints"
N_CHANNELS = 63
N_TIMES = 250


class ConvNormAct(nn.Module):
    def __init__(self, c_in, c_out, k=5, stride=1, transpose=False, groups=1):
        super().__init__()
        if transpose:
            # Stride-2 transpose conv that exactly doubles length when k=4, p=1.
            assert k == 4 and stride == 2, "transpose ConvBlock assumes k=4 stride=2"
            self.conv = nn.ConvTranspose1d(c_in, c_out, kernel_size=k,
                                           stride=stride, padding=1,
                                           output_padding=0, groups=groups)
        else:
            self.conv = nn.Conv1d(c_in, c_out, k, stride=stride,
                                  padding=k // 2, groups=groups)
        self.norm = nn.GroupNorm(min(8, c_out), c_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block over a sequence of tokens.

    Used at the codec bottleneck: conv stack gives local temporal features
    fast; attention then mixes information across the full latent timeline.
    Local + global is the well-trodden ViT-for-1D pattern.
    """

    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        h = self.n1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x


class TransformerStack(nn.Module):
    """N transformer blocks with sinusoidal positional encoding.

    Operates on conv-tower outputs reshaped as a sequence of T tokens of dim D
    (D = conv hidden_size * 2). The latent's *temporal* dimension is the
    sequence axis; cross-channel mixing inside each token is done by the MLP.
    """

    def __init__(self, dim: int, n_layers: int, max_len: int = 64,
                 n_heads: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads=n_heads) for _ in range(n_layers)
        ])
        # Sinusoidal PE — fixed, no params.
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:
        # x_btd: (B, T, D)
        x = x_btd + self.pe[: x_btd.size(1)].unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        return x


class EEGEncoder(nn.Module):
    """(B, 63, 250) -> (B, C_lat, T_lat). Default: 32 ch x 32 ts.

    Optional `n_attn` transformer blocks at the bottleneck (after conv
    downsampling, before latent projection). With n_attn=0 the architecture
    is byte-identical to the original conv-only codec (old checkpoints load).
    """

    def __init__(self, c_lat: int = 32, hidden: int = 128, n_attn: int = 0,
                 attn_heads: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_CHANNELS, hidden, kernel_size=15, padding=7),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        # 250 -> pad to 256 -> /2 /2 /2 -> 32 latent timesteps.
        self.body = nn.Sequential(
            ConvNormAct(hidden, hidden, k=5, stride=2),       # 128
            ConvNormAct(hidden, hidden * 2, k=5, stride=2),   # 64
            ConvNormAct(hidden * 2, hidden * 2, k=3, stride=2),  # 32
        )
        self.n_attn = n_attn
        if n_attn > 0:
            self.attn = TransformerStack(dim=hidden * 2, n_layers=n_attn,
                                          max_len=64, n_heads=attn_heads)
        self.to_latent = nn.Conv1d(hidden * 2, c_lat, kernel_size=1)
        self.c_lat = c_lat

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        eeg = F.pad(eeg, (3, 3), mode="reflect")
        x = self.stem(eeg)
        x = self.body(x)                                    # (B, 2H, 32)
        if self.n_attn > 0:
            x_seq = x.transpose(1, 2)                       # (B, 32, 2H)
            x_seq = self.attn(x_seq)
            x = x_seq.transpose(1, 2)
        return self.to_latent(x)                            # (B, c_lat, 32)


class EEGDecoder(nn.Module):
    """(B, C_lat, 32) -> (B, 63, 250). Mirror of EEGEncoder."""

    def __init__(self, c_lat: int = 32, hidden: int = 128, n_attn: int = 0,
                 attn_heads: int = 4):
        super().__init__()
        self.from_latent = nn.Conv1d(c_lat, hidden * 2, kernel_size=1)
        self.n_attn = n_attn
        if n_attn > 0:
            self.attn = TransformerStack(dim=hidden * 2, n_layers=n_attn,
                                          max_len=64, n_heads=attn_heads)
        self.body = nn.Sequential(
            ConvNormAct(hidden * 2, hidden * 2, k=4, stride=2, transpose=True),  # 64
            ConvNormAct(hidden * 2, hidden, k=4, stride=2, transpose=True),      # 128
            ConvNormAct(hidden, hidden, k=4, stride=2, transpose=True),          # 256
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, N_CHANNELS, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z)
        if self.n_attn > 0:
            x_seq = x.transpose(1, 2)
            x_seq = self.attn(x_seq)
            x = x_seq.transpose(1, 2)
        x = self.body(x)
        x = self.head(x)
        return x[..., 3:-3]


class FactorizedLaplacePrior(nn.Module):
    """Per-channel Laplace prior with learnable scale.

    bits_per_symbol(y) = -log2 ( CDF(y + 0.5) - CDF(y - 0.5) )
    For Laplace(0, b): CDF(z) = 0.5 + 0.5 * sign(z) * (1 - exp(-|z|/b))

    During training we pass the noisy latent ~ y + U(-0.5, 0.5) and treat it
    as the integer-rounded symbol -- the average bits estimate is unbiased
    when y is uniformly distributed within a bin (Balle et al. 2017).
    """

    def __init__(self, c_lat: int):
        super().__init__()
        # Reparameterize scale via softplus so it stays positive.
        self.log_b = nn.Parameter(torch.zeros(c_lat))

    def _cdf(self, z: torch.Tensor) -> torch.Tensor:
        b = F.softplus(self.log_b)[None, :, None].clamp_min(1e-4)
        return 0.5 + 0.5 * torch.sign(z) * (1.0 - torch.exp(-z.abs() / b))

    def bits(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, C, T) integer (or noisy) latent. Returns scalar mean bits/symbol."""
        upper = self._cdf(y + 0.5)
        lower = self._cdf(y - 0.5)
        p = (upper - lower).clamp_min(1e-9)
        return -torch.log2(p).mean()


def quantize(y: torch.Tensor, training: bool) -> torch.Tensor:
    """One of the 3 easy-to-get-wrong spots.

    Train: additive uniform noise as a stochastic, differentiable proxy for
    rounding. Inference: real integer round (use straight-through so caller
    code is identical, gradient is the identity)."""
    if training:
        return y + (torch.rand_like(y) - 0.5)
    # Straight-through round for inference (no gradients here anyway).
    return torch.round(y).detach() + (y - y.detach())


class EEGCodec(nn.Module):
    def __init__(self, c_lat: int = 32, hidden: int = 128, n_attn: int = 0,
                 attn_heads: int = 4):
        super().__init__()
        self.encoder = EEGEncoder(c_lat=c_lat, hidden=hidden,
                                   n_attn=n_attn, attn_heads=attn_heads)
        self.decoder = EEGDecoder(c_lat=c_lat, hidden=hidden,
                                   n_attn=n_attn, attn_heads=attn_heads)
        self.prior = FactorizedLaplacePrior(c_lat=c_lat)
        self.c_lat = c_lat
        self.n_attn = n_attn

    def forward(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.encoder(eeg)
        y_q = quantize(y, self.training)
        bits_per_symbol = self.prior.bits(y_q)
        eeg_hat = self.decoder(y_q)
        return eeg_hat, bits_per_symbol, y_q

    @torch.no_grad()
    def compress_then_reconstruct(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference path: real rounding, real bits/symbol."""
        was_training = self.training
        self.eval()
        y = self.encoder(eeg)
        y_int = torch.round(y)
        bits = self.prior.bits(y_int)
        eeg_hat = self.decoder(y_int)
        if was_training:
            self.train()
        return eeg_hat, bits

    # ----- helpers for reporting -----
    def latent_shape(self) -> tuple[int, int]:
        return (self.c_lat, 32)

    def bpp_floor(self, bits_per_symbol: float) -> float:
        """Bits per (real) EEG sample = (bits/symbol * #symbols) / #samples."""
        c, t = self.latent_shape()
        n_symbols = c * t
        n_samples = N_CHANNELS * N_TIMES
        return bits_per_symbol * n_symbols / n_samples


def compression_ratio(model: EEGCodec, bits_per_symbol: float) -> float:
    """Reference baseline = float16 (16 bits/sample). Ratio = 16 / bpp."""
    bpp = model.bpp_floor(bits_per_symbol)
    return 16.0 / max(bpp, 1e-6)
