"""NeuroZipLayer - a frozen, drop-in EEG embedding layer.

Instead of a trainable embedding at the front of your model, drop in NeuroZipLayer:
the trained, frozen EEG->CLIP-space projector. It maps a normalized EEG epoch
(B, 63, 250) to a unit-norm 512-d semantic vector. Frozen, so you never spend
gradients on it; gradients still pass through to the rest of your network.

    import torch
    from neurozip import NeuroZipLayer

    embed = NeuroZipLayer.from_pretrained()   # loads the trained clip_proj.pt, frozen
    x = torch.randn(8, 63, 250)
    z = embed(x)                              # (8, 512), no grad spent on `embed`
    logits = my_head(z)                       # train only my_head

Requires the `ml` extra (pip install "neurozip[ml]") and the model checkpoints
(local ./checkpoints or `neurozip download`).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "NeuroZipLayer needs PyTorch. Install the ML extra:  pip install 'neurozip[ml]'"
    ) from exc

from .models.clip_proj import EEGProjector


class NeuroZipLayer(nn.Module):
    """Frozen EEG -> 512-d CLIP-space embedding, usable as a model's first layer."""

    def __init__(self, projector: EEGProjector):
        super().__init__()
        self.projector = projector
        self.freeze()

    def freeze(self) -> "NeuroZipLayer":
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    def forward(self, eeg: "torch.Tensor") -> "torch.Tensor":
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        return self.projector(eeg)            # already L2-normalized

    @classmethod
    def from_pretrained(cls, path=None) -> "NeuroZipLayer":
        """Load the trained frozen projector (clip_proj.pt).

        `path` may point at a specific clip_proj.pt; otherwise it is resolved from
        $NEUROZIP_HOME / ./checkpoints / ~/.neurozip (see runtime.checkpoints_dir).
        """
        from .models.clip_proj import load_frozen_projector
        from . import runtime
        if path is None:
            path = runtime.checkpoints_dir() / "clip_proj.pt"
        proj = load_frozen_projector(path, device="cpu")
        return cls(proj)
