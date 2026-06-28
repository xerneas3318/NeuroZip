"""Real inference runtime: load checkpoints, compress / decompress / embed EEG.

Everything here needs the `ml` extra (torch). Imports are done inside functions so
`import neurozip` and the CLI stay dependency-free until a real command runs.

Models operate in NORMALIZED EEG space (per-channel z-scored with the training
norm stats) at 63 channels x 250 timepoints, which is what the codecs were trained on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

N_CHANNELS = 63
N_TIMES = 250
TIERS = ("low", "med", "high", "xhigh")
_PKG = Path(__file__).resolve().parent

# Trained model bundle (projector + v4 codecs + scores + norm stats + demo samples).
MODELS_URL = ("https://github.com/xerneas3318/NeuroZip/releases/download/"
              "v0.2.0-beta.1/neurozip-models-v4.tar.gz")
# Source tarball used to self-bootstrap the ML environment (must match this version).
SOURCE_URL = ("https://github.com/xerneas3318/NeuroZip/releases/download/"
              "v0.2.0-beta.4/neurozip-0.2.0-beta.4.tar.gz")


def _progress(label):
    import sys
    state = {"last": -1}

    def hook(blocks, bs, total):
        done = blocks * bs
        if total > 0:
            pct = min(100, int(done * 100 / total))
            if pct != state["last"]:
                state["last"] = pct
                bar = "#" * (pct // 4) + "-" * (25 - pct // 4)
                sys.stdout.write(f"\r[neurozip] {label} [{bar}] {pct:3d}% "
                                 f"({done // 1048576}/{total // 1048576} MB)")
                sys.stdout.flush()
    return hook


def download_models(url: str | None = None, dest: str | None = None) -> Path:
    """Fetch and extract the model bundle into <dest>/checkpoints (default ~/.neurozip)."""
    import sys
    import tarfile
    import tempfile
    import urllib.request
    url = url or MODELS_URL
    home = Path(dest) if dest else (Path.home() / ".neurozip")
    home.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name, reporthook=_progress("downloading models"))
        sys.stdout.write("\n")
        with tarfile.open(tmp.name, "r:gz") as tf:
            tf.extractall(home)
    ck = home / "checkpoints"
    print(f"[neurozip] models ready at {ck}")
    return ck


def ensure_models(ckpt: Path | None = None) -> Path:
    """Return the checkpoints dir, downloading the bundle on first use if absent."""
    ckpt = ckpt or checkpoints_dir()
    if (ckpt / "clip_proj.pt").exists():
        return ckpt
    return download_models()


def checkpoints_dir() -> Path:
    """Resolve where the model checkpoints live, first hit wins.

    1. $NEUROZIP_HOME/checkpoints
    2. ./checkpoints (current working dir)
    3. <repo>/checkpoints (next to the package)
    4. ~/.neurozip/checkpoints   (where `neurozip download` puts them)
    """
    candidates = []
    if os.environ.get("NEUROZIP_HOME"):
        candidates.append(Path(os.environ["NEUROZIP_HOME"]) / "checkpoints")
    candidates += [
        Path.cwd() / "checkpoints",
        _PKG.parent / "checkpoints",
        Path.home() / ".neurozip" / "checkpoints",
    ]
    for c in candidates:
        if (c / "clip_proj.pt").exists():
            return c
    # Default target for downloads even if not present yet.
    return Path.home() / ".neurozip" / "checkpoints"


def _require_models(ckpt: Path):
    if not (ckpt / "clip_proj.pt").exists():
        raise FileNotFoundError(
            f"No checkpoints found (looked in {ckpt}). Run `neurozip download` "
            "to fetch the models, or set $NEUROZIP_HOME."
        )


def _torch():
    try:
        import torch  # noqa
        return torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Real inference needs PyTorch. Install the ML extra:  pip install 'neurozip[ml]'"
        ) from exc


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_norm_stats(ckpt: Path | None = None):
    torch = _torch()
    ckpt = ckpt or checkpoints_dir()
    s = torch.load(ckpt / "norm_stats.pt", weights_only=False, map_location="cpu")
    return s["mean"], s["std"]            # each (63,)


def load_codec(tier: str = "high", variant: str = "neurozip", ckpt: Path | None = None):
    torch = _torch()
    from .models.codec import EEGCodec
    ckpt = ckpt or checkpoints_dir()
    _require_models(ckpt)
    path = ckpt / f"{variant}_v4_{tier}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing codec checkpoint {path.name} in {ckpt}.")
    state = torch.load(path, weights_only=False, map_location="cpu")
    cfg = state.get("config", {})
    c = EEGCodec(c_lat=cfg.get("c_lat", 32), hidden=cfg.get("hidden", 128),
                 n_attn=cfg.get("n_attn", 0))
    c.load_state_dict(state["model"])
    c.eval()
    return c


def load_projector(ckpt: Path | None = None):
    from .models.clip_proj import load_frozen_projector
    ckpt = ckpt or checkpoints_dir()
    _require_models(ckpt)
    return load_frozen_projector(ckpt / "clip_proj.pt", device="cpu")


def scores(ckpt: Path | None = None) -> dict:
    ckpt = ckpt or checkpoints_dir()
    p = ckpt / "scores.json"
    return json.loads(p.read_text()) if p.exists() else {}


# --------------------------------------------------------------------------- #
# Core ops (all in normalized EEG space; arrays are numpy or torch)
# --------------------------------------------------------------------------- #

def _as_batch(torch, eeg):
    """Accept (63,250) or (N,63,250) numpy/torch -> (N,63,250) float tensor."""
    import numpy as np
    if isinstance(eeg, np.ndarray):
        eeg = torch.from_numpy(eeg)
    eeg = eeg.float()
    if eeg.dim() == 2:
        eeg = eeg.unsqueeze(0)
    assert eeg.shape[-2:] == (N_CHANNELS, N_TIMES), \
        f"expected (...,{N_CHANNELS},{N_TIMES}), got {tuple(eeg.shape)}"
    return eeg


def compress(eeg, tier: str = "high", variant: str = "neurozip", codec=None):
    """Encode normalized EEG -> integer latents + real rate stats.

    Returns a dict: latents (int16 numpy), bits_per_symbol, bpp, ratio_vs_fp16,
    latent_shape, tier, variant, n_epochs.
    """
    torch = _torch()
    import numpy as np
    codec = codec or load_codec(tier, variant)
    x = _as_batch(torch, eeg)
    with torch.no_grad():
        y = codec.encoder(x)                       # (N, c_lat, 32)
        y_int = torch.round(y)
        bits = codec.prior.bits(y_int).item()
    bpp = codec.bpp_floor(bits)
    return {
        "latents": y_int.to(torch.int16).cpu().numpy(),
        "bits_per_symbol": float(bits),
        "bpp": float(bpp),
        "ratio_vs_fp16": float(16.0 / max(bpp, 1e-6)),
        "latent_shape": list(y_int.shape[1:]),
        "tier": tier,
        "variant": variant,
        "n_epochs": int(x.shape[0]),
    }


def decompress(latents, tier: str = "high", variant: str = "neurozip", codec=None):
    """Decode integer latents -> reconstructed normalized EEG (numpy (N,63,250))."""
    torch = _torch()
    import numpy as np
    codec = codec or load_codec(tier, variant)
    z = latents if hasattr(latents, "shape") else np.asarray(latents)
    z = torch.from_numpy(np.asarray(z)).float()
    if z.dim() == 2:
        z = z.unsqueeze(0)
    with torch.no_grad():
        xh = codec.decoder(z)
    return xh.cpu().numpy()


def embed(eeg, projector=None):
    """Normalized EEG -> (N,512) CLIP-space embeddings from the frozen judge."""
    torch = _torch()
    projector = projector or load_projector()
    x = _as_batch(torch, eeg)
    with torch.no_grad():
        z = projector(x)
    return z.cpu().numpy()


def reconstruct(eeg, tier: str = "high", variant: str = "neurozip", codec=None):
    """Convenience: compress then decompress, returning (recon, stats, mse)."""
    torch = _torch()
    codec = codec or load_codec(tier, variant)
    stats = compress(eeg, tier, variant, codec=codec)
    recon = decompress(stats["latents"], tier, variant, codec=codec)
    x = _as_batch(torch, eeg).cpu().numpy()
    mse = float(((x - recon) ** 2).mean())
    stats["mse"] = mse
    return recon, stats, mse


def denormalize(eeg_norm, ckpt: Path | None = None):
    """Map normalized EEG back to whitened units using the saved stats."""
    torch = _torch()
    import numpy as np
    mean, std = load_norm_stats(ckpt)
    mean = mean.numpy()[:, None]
    std = std.numpy()[:, None]
    arr = np.asarray(eeg_norm)
    return arr * std + mean


# --------------------------------------------------------------------------- #
# .nz archive I/O  (a compressed npz: int latents + metadata)
# --------------------------------------------------------------------------- #

def save_nz(path: str, stats: dict):
    import numpy as np
    meta = {k: v for k, v in stats.items() if k != "latents"}
    # Write through a handle so numpy keeps our exact `.nz` name (it would
    # otherwise append `.npz`).
    with open(path, "wb") as fh:
        np.savez_compressed(fh, latents=stats["latents"], meta=json.dumps(meta))


def load_nz(path: str) -> dict:
    import numpy as np
    d = np.load(path, allow_pickle=True)
    out = json.loads(str(d["meta"]))
    out["latents"] = d["latents"]
    return out


def load_samples(ckpt: Path | None = None):
    """Return (eeg (N,63,250) float32, concepts list) bundled for the demo, or None."""
    import numpy as np
    ckpt = ckpt or checkpoints_dir()
    p = ckpt / "samples.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d["eeg"].astype("float32"), [str(c) for c in d["concepts"]]
