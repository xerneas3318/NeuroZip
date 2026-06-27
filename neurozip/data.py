"""Stage 0 — data loading and sanity (THINGS-EEG).

Loads `Haitao999/things-eeg`, exposing (eeg, image, concept_text) per example.

GOTCHA (3): keep the per-channel normalization stats you compute here — they are
required to (a) report a real bitrate and (b) invert the EEG on decompression.
GOTCHA: EEG is stored float16 — cast to float32 before training.
GOTCHA: concept labels carry a numeric prefix (`000001_aardvark`) — strip it.
"""

from __future__ import annotations

DATASET_ID = "Haitao999/things-eeg"


def clean_concept(label: str) -> str:
    """`000001_aardvark` -> `aardvark`."""
    return label.split("_", 1)[1] if "_" in label else label


def summarize():
    """Stage-0 stand-alone deliverable: load, print a summary table, confirm shapes.

    Intentionally not implemented yet — see instructions.md (gitignored) Stage 0.
    Print: examples per split, EEG epoch shape, channels, timepoints, value range,
    number of distinct concepts. Do NOT assume shapes — print them.
    """
    raise NotImplementedError("Stage 0: implement THINGS-EEG load + shape summary.")
