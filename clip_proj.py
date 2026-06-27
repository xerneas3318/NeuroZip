"""EEG -> CLIP projector P.

Stage 1 of NeuroZip: the **frozen judge**. P is trained to map a single-trial EEG
epoch (63 ch x 250 ts) into CLIP's image embedding space (dim 512), so that
P(eeg) is close to CLIP(seen_image). Standard THINGS-EEG alignment, only the
EEG side is trained (CLIP is precomputed and never invoked here).

Once trained, P is FROZEN for Stages 3-5 and acts as the differentiable judge
inside the codec's task loss.

Architecture: depthwise temporal conv -> spatial mix (1x1) -> conv stack with
GroupNorm/GELU -> global-avg pool over time -> small MLP -> L2-normalize.
Small enough to train in a couple of minutes on a 4090.
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


class EEGProjector(nn.Module):
    """Single-trial EEG -> CLIP image embedding (unit-norm)."""

    def __init__(self, n_channels: int = 63, hidden: int = 128, out_dim: int = CLIP_DIM):
        super().__init__()
        # Stage A: temporal conv per channel (depthwise), then channel mix.
        self.temporal = nn.Conv1d(n_channels, n_channels, kernel_size=15,
                                  padding=7, groups=n_channels)
        self.spatial = nn.Conv1d(n_channels, hidden, kernel_size=1)
        self.tnorm = nn.GroupNorm(8, hidden)
        # Stage B: downsampling conv tower (250 -> ~16 timesteps).
        self.tower = nn.Sequential(
            ConvBlock(hidden, hidden, k=7, stride=2),   # 125
            ConvBlock(hidden, hidden * 2, k=5, stride=2),   # 63
            ConvBlock(hidden * 2, hidden * 2, k=5, stride=2),  # 32
            ConvBlock(hidden * 2, hidden * 2, k=3, stride=2),  # 16
        )
        # Stage C: head.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden * 2, hidden * 4), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden * 4, out_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: (B, 63, 250)
        x = self.temporal(eeg)
        x = self.tnorm(self.spatial(x))
        x = F.gelu(x)
        x = self.tower(x)
        x = self.head(x)
        return F.normalize(x, dim=-1)


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
                         out_dim=cfg.get("out_dim", CLIP_DIM)).to(device)
        p.load_state_dict(state["model"])
    else:                                            # legacy raw state_dict
        # Infer hidden from spatial.weight shape: (hidden, 63, 1).
        hidden = int(state["spatial.weight"].shape[0])
        p = EEGProjector(hidden=hidden).to(device)
        p.load_state_dict(state)
    p.eval()
    for prm in p.parameters():
        prm.requires_grad_(False)
    return p
