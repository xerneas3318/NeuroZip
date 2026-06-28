"""Stage 1 — frozen CLIP + EEG->CLIP projector (the judge).

CLIP is loaded pretrained and frozen forever. We train ONLY a small EEG->CLIP-space
projector `P` with an InfoNCE loss aligning P(eeg) to the frozen CLIP *image* embedding
of the paired seen image. Because CLIP's text/image encoders share a space, text
retrieval works for free at inference. After training, `P` is frozen too and becomes
part of the immutable judge used by the codec's task loss (Stage 3).
"""

from __future__ import annotations


def build_judge():
    """Return (clip_model, eeg_projector), both eventually frozen.

    Not implemented yet — see instructions.md Stage 1.
    """
    raise NotImplementedError("Stage 1: build frozen CLIP + trainable EEG projector.")


def train_projector():
    """Train P (InfoNCE EEG<->CLIP-image), report top-1/top-5 retrieval, save weights."""
    raise NotImplementedError("Stage 1: train EEG->CLIP projector.")
