"""Beta placeholder backend (stdlib-only).

Real compression lands in `codec.py` (needs the `ml` extra). Until then these functions
emit plausible *randomized* results and write a tiny self-describing `.nz` sidecar so the
CLI and the local UI can be exercised end to end. Everything here is intentionally
dependency-free so it works in a fast `brew install`.
"""

from __future__ import annotations

import json
import os
import random
import time

NZ_MAGIC = "NEUROZIP-BETA"


def _iter_files(path: str):
    if os.path.isfile(path):
        yield path
        return
    for root, _dirs, files in os.walk(path):
        for f in files:
            yield os.path.join(root, f)


def _bytes_of(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def compress(input_path: str, output_path: str, ratio: float = 50.0) -> dict:
    """Placeholder compress: tally real sizes, invent a compressed size near `ratio`."""
    files = list(_iter_files(input_path))
    original = sum(_bytes_of(f) for f in files) or random.randint(2_000_000, 9_000_000)
    # jitter the achieved ratio a little around the target so it feels real
    achieved = max(1.0, random.uniform(ratio * 0.8, ratio * 1.15))
    compressed = max(1, int(original / achieved))
    manifest = {
        "magic": NZ_MAGIC,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": os.path.abspath(input_path),
        "n_items": len(files) or 1,
        "original_bytes": original,
        "compressed_bytes": compressed,
        "ratio": round(original / compressed, 2),
        "target_ratio": ratio,
        "note": "beta placeholder archive — no real signal data stored",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def decompress(archive_path: str, output_dir: str) -> dict:
    """Placeholder decompress: read the sidecar, fabricate a restored payload."""
    try:
        with open(archive_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError):
        manifest = {"n_items": 1, "original_bytes": random.randint(2_000_000, 9_000_000)}
    os.makedirs(output_dir, exist_ok=True)
    restored = manifest.get("original_bytes", random.randint(2_000_000, 9_000_000))
    with open(os.path.join(output_dir, "RESTORED.txt"), "w", encoding="utf-8") as fh:
        fh.write("beta placeholder — real epochs will be reconstructed here.\n")
    return {"n_items": manifest.get("n_items", 1), "restored_bytes": restored}


_DEMO_CONCEPTS = [
    "accordion", "aardvark", "lighthouse", "violin", "pineapple", "compass",
    "telescope", "jellyfish", "harmonica", "lantern", "umbrella", "cactus",
]


def search(query: str, archive_path: str, topk: int = 5) -> list[tuple[str, float]]:
    """Placeholder retrieval: random concepts with descending random scores."""
    pool = [query] + [c for c in _DEMO_CONCEPTS if c != query]
    random.shuffle(pool)
    chosen = pool[: max(1, topk)]
    scores = sorted((random.uniform(0.35, 0.95) for _ in chosen), reverse=True)
    return list(zip(chosen, scores))
