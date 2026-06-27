"""NeuroZipLayer — a frozen, drop-in embedding layer.

The whole point: instead of putting a trainable embedding layer at the front of your
model and backpropagating through it, drop in NeuroZipLayer. It is a pretrained,
*frozen* encoder (the same judge NeuroZip compresses against) that maps a raw signal
epoch to a semantic embedding. Gradients pass straight through to the rest of your
network — you train everything downstream, but never waste compute updating the
embedding.

    import torch
    from neurozip import NeuroZipLayer

    embed = NeuroZipLayer.from_pretrained("eeg-clip-b32")   # frozen
    x = torch.randn(8, 63, 200)            # (batch, channels, time)
    z = embed(x)                           # (8, embed_dim) — no grad spent on `embed`
    logits = my_head(z)                    # train only `my_head`

Requires the `ml` extra:  pip install "neurozip[ml]"
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "NeuroZipLayer needs PyTorch. Install the ML extra:  pip install 'neurozip[ml]'"
    ) from exc


class NeuroZipLayer(nn.Module):
    """Frozen signal -> embedding encoder usable as the first layer of any model.

    Beta: weights are randomly initialized but frozen, so the API and the
    no-grad-on-embedding training pattern are exactly what they will be once a real
    checkpoint is published via `from_pretrained`.
    """

    def __init__(self, in_channels: int = 63, embed_dim: int = 512, freeze: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.project = nn.Linear(256, embed_dim)
        if freeze:
            self.freeze()

    def freeze(self) -> "NeuroZipLayer":
        """Freeze all parameters (default). The layer still passes gradients through."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, channels, time) -> (batch, embed_dim), L2-normalized like CLIP.
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.encoder(x).squeeze(-1)
        z = self.project(h)
        return torch.nn.functional.normalize(z, dim=-1)

    @classmethod
    def from_pretrained(cls, name_or_path: str = "eeg-clip-b32", **kwargs) -> "NeuroZipLayer":
        """Load a published frozen checkpoint.

        Beta: no checkpoints are published yet, so this returns a frozen
        randomly-initialized layer and warns. The call signature is stable.
        """
        layer = cls(**kwargs)
        looks_like_path = "/" in name_or_path or name_or_path.endswith((".pt", ".pth"))
        if looks_like_path:
            state = torch.load(name_or_path, map_location="cpu")
            layer.load_state_dict(state)
        else:
            import warnings

            warnings.warn(
                f"NeuroZipLayer.from_pretrained({name_or_path!r}): no published checkpoint "
                "in this beta; returning a frozen randomly-initialized layer.",
                stacklevel=2,
            )
        return layer.freeze()
