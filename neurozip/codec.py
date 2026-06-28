"""Stage 2 + 3 — the compressor and the task-aware loss.

Stage 2: a 1D-conv (channel-aware) autoencoder + factorized entropy model for a real
bits-per-epoch estimate. Trained with rate + reconstruction MSE only (the fidelity-only
baseline).

GOTCHA (2): quantizer is noise-at-train / round-at-inference. At training, add uniform
noise in [-0.5, 0.5] as a differentiable proxy for rounding; at inference use round().

Stage 3 (the novel part): add the task term
    L = rate + lambda_recon*MSE(eeg, dec) + lambda_task*(1 - cos(P(dec), CLIP_img(seen)))

GOTCHA (1): frozen-but-NOT-detached gradient flow. P and CLIP have requires_grad_(False),
but do NOT wrap P(dec) in no_grad — gradients must flow through the frozen judge back into
the decoder. Only the *target* CLIP_img(seen) goes under no_grad. After a task-only
backward, assert a decoder param's .grad is non-None and non-zero.
"""

from __future__ import annotations


class EEGCodec:
    """Autoencoder codec with a learned entropy model. Not implemented yet."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Stage 2: implement EEG autoencoder codec.")


def task_aware_loss(*args, **kwargs):
    """Stage 3 task-aware loss. Not implemented yet — see instructions.md Stage 3."""
    raise NotImplementedError("Stage 3: implement rate + recon + task loss.")
