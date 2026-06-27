"""Stage 4 — evaluation + demo artifacts.

At matched compression ratios, compare NeuroZip vs the fidelity-only codec on:
  - bpp / compression ratio
  - EEG reconstruction error (NeuroZip should be WORSE here — that's expected)
  - retained retrieval accuracy (top-k text->EEG and image->EEG via P)
  - HELD-OUT circularity defense: retained accuracy via a separate concept classifier
    that the compressor's loss never saw. Build this in — a judge will ask.

Produces the money plot (rate vs retained retrieval) and precomputed demo assets
(JSON + paired stimulus PNGs) consumed by ui/demo.html.
"""

from __future__ import annotations


def main():
    raise NotImplementedError("Stage 4: implement evaluation + asset generation.")


if __name__ == "__main__":
    main()
