"""EEG -> CLIP projector P.

Stage 1 of NeuroZip: the **frozen judge**. P maps a single-trial EEG epoch
(63 ch x 250 ts) into CLIP's image embedding space (dim 512). CLIP itself is
never invoked here; we contrast against the dataset's precomputed image
features.

With `n_attn > 0`, the global-avg-pool head is replaced by a [CLS]-token
transformer head over the conv tower's 16 temporal tokens. This lets the
projector compare ERP windows (P100 vs P300, etc.) in a single operation
rather than relying on the conv stack's stacked receptive field.
"""

from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_DIM = 512
WEIGHTS = Path(__file__).parent / "checkpoints" / "clip_proj.pt"


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, stride=stride,
                              padding=k // 2, groups=groups)
        self.norm = nn.GroupNorm(min(8, c_out), c_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _TfmBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x


class EEGProjector(nn.Module):
    """Single-trial EEG -> CLIP image embedding (unit-norm).

    With n_attn=0: original conv-tower + AvgPool + MLP head (backward-compat).
    With n_attn>0: conv-tower output (B, 2H, 16) -> prepend [CLS] token ->
                    n_attn transformer blocks -> [CLS] vector -> Linear -> L2.
    """

    def __init__(self, n_channels: int = 63, hidden: int = 128,
                 out_dim: int = CLIP_DIM, n_attn: int = 0,
                 attn_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_attn = n_attn
        self.hidden = hidden
        # Stage A: temporal conv per channel (depthwise), then channel mix.
        self.temporal = nn.Conv1d(n_channels, n_channels, kernel_size=15,
                                  padding=7, groups=n_channels)
        self.spatial = nn.Conv1d(n_channels, hidden, kernel_size=1)
        self.tnorm = nn.GroupNorm(8, hidden)
        # Stage B: downsampling conv tower (250 -> 16 timesteps).
        self.tower = nn.Sequential(
            ConvBlock(hidden, hidden, k=7, stride=2),         # 125
            ConvBlock(hidden, hidden * 2, k=5, stride=2),     # 63
            ConvBlock(hidden * 2, hidden * 2, k=5, stride=2), # 32
            ConvBlock(hidden * 2, hidden * 2, k=3, stride=2), # 16
        )
        token_dim = hidden * 2

        if n_attn > 0:
            # Stage C-attn: [CLS] + transformer over 16 temporal tokens.
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            # Sinusoidal PE for tokens (length 17 = CLS + 16 conv tokens)
            pe = torch.zeros(17, token_dim)
            pos = torch.arange(17).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, token_dim, 2).float() * (-math.log(10000.0) / token_dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe)
            self.tfm = nn.ModuleList([
                _TfmBlock(token_dim, n_heads=attn_heads, dropout=dropout * 0.5)
                for _ in range(n_attn)
            ])
            self.cls_norm = nn.LayerNorm(token_dim)
            self.cls_head = nn.Sequential(
                nn.Linear(token_dim, hidden * 4), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 4, out_dim),
            )
        else:
            # Stage C-original: AvgPool + MLP head (preserved for v1 checkpoints).
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(token_dim, hidden * 4), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden * 4, out_dim),
            )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: (B, 63, 250)
        x = self.temporal(eeg)
        x = self.tnorm(self.spatial(x))
        x = F.gelu(x)
        x = self.tower(x)                          # (B, 2H, 16)

        if self.n_attn > 0:
            tokens = x.transpose(1, 2)             # (B, 16, 2H)
            B = tokens.size(0)
            cls = self.cls_token.expand(B, -1, -1) # (B, 1, 2H)
            seq = torch.cat([cls, tokens], dim=1)  # (B, 17, 2H)
            seq = seq + self.pe.unsqueeze(0)
            for blk in self.tfm:
                seq = blk(seq)
            cls_out = self.cls_norm(seq[:, 0])     # (B, 2H)
            out = self.cls_head(cls_out)
        else:
            out = self.head(x)

        return F.normalize(out, dim=-1)


def info_nce(eeg_emb: torch.Tensor, clip_emb: torch.Tensor,
             temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE between batches of L2-normalized embeddings."""
    # Both already L2-normalized (P does it; we normalize CLIP here too).
    clip_emb = F.normalize(clip_emb, dim=-1)
    logits = (eeg_emb @ clip_emb.t()) / temperature
    targets = torch.arange(eeg_emb.size(0), device=eeg_emb.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


@torch.no_grad()
def retrieval_topk(eeg_emb: torch.Tensor, gallery: torch.Tensor,
                   gold_idx: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """eeg_emb (Q, D), gallery (G, D), gold_idx (Q,)."""
    eeg_emb = F.normalize(eeg_emb, dim=-1)
    gallery = F.normalize(gallery, dim=-1)
    sims = eeg_emb @ gallery.t()                       # (Q, G)
    ranks = sims.argsort(dim=-1, descending=True)
    out = {}
    for k in ks:
        topk = ranks[:, :k]
        out[f"top{k}"] = (topk == gold_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    return out


def load_frozen_projector(path: Path = WEIGHTS, device: str = "cpu") -> EEGProjector:
    """Load P for use as a frozen judge in later stages."""
    state = torch.load(path, weights_only=False, map_location=device)
    if isinstance(state, dict) and "model" in state:
        cfg = state.get("config", {})
        p = EEGProjector(hidden=cfg.get("hidden", 128),
                         out_dim=cfg.get("out_dim", CLIP_DIM),
                         n_attn=cfg.get("n_attn", 0),
                         attn_heads=cfg.get("attn_heads", 4)).to(device)
        p.load_state_dict(state["model"])
    else:                                            # legacy raw state_dict
        hidden = int(state["spatial.weight"].shape[0])
        p = EEGProjector(hidden=hidden).to(device)
        p.load_state_dict(state)
    p.eval()
    for prm in p.parameters():
        prm.requires_grad_(False)
    return p
