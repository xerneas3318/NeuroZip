"""Shared harness for the rian-2 codec experiments (Group A).

These experiments probe *mechanisms* of the NeuroZip codec on a controlled,
structured surrogate for THINGS-EEG (the real 3 GB dataset / trained
checkpoints are not needed to answer the questions here, and this keeps CPU/
network light). Everything below uses the REAL codec/projector modules from
the repo root (``codec.py`` / ``clip_proj.py``), so the entropy model, the
quantizer, and the projector are exactly the ones NeuroZip ships.

Surrogate design (so the questions are actually answerable):

* ``N_CONCEPTS`` object concepts. Each has (a) a random unit "prototype" in
  R^512 standing in for the CLIP image embedding, and (b) a characteristic
  spatiotemporal EEG signature built from ERP-like Gaussian bumps + a couple
  of oscillations, mixed through a *shared* R-source spatial mixing matrix so
  channels are genuinely correlated (this is what a hyperprior / mixture / VQ
  can exploit, and what a factorized prior cannot).
* A trial = concept signature + per-trial Gaussian noise, then per-channel
  z-normalized exactly like ``data.py``.

The point is a controlled comparison where only the studied component changes.
Numbers are directional (small models, few epochs); the mechanism is real.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

# Import the real repo modules (repo root is the parent of tests/).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_CH = 63
N_T = 250
CLIP_DIM = 512


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Structured synthetic EEG surrogate
# --------------------------------------------------------------------------- #

def _concept_bank(n_concepts: int, n_sources: int, seed: int):
    """Return (templates (C,63,250) float32, prototypes (C,512) unit-norm)."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_T)
    # ERP-like Gaussian bumps at plausible latencies (in samples @ 250 Hz).
    lat = [25, 45, 75, 100, 130, 165, 200]
    wid = [7, 9, 11, 13, 16, 14, 20]
    bumps = np.stack([np.exp(-0.5 * ((t - l) / w) ** 2) for l, w in zip(lat, wid)])
    osc = np.stack([np.sin(2 * np.pi * f * t / 250.0) for f in (6.0, 10.0)])
    B = np.concatenate([bumps, osc], axis=0)              # (K, 250)
    K = B.shape[0]

    # Shared spatial mixing: R latent sources -> 63 channels (fixed across
    # concepts) => cross-channel correlation the priors can model.
    A = rng.normal(0, 1, size=(N_CH, n_sources)).astype(np.float32)
    # Concept-specific source x basis coefficients.
    G = rng.normal(0, 1, size=(n_concepts, n_sources, K)).astype(np.float32)
    source_time = np.einsum("crk,kt->crt", G, B.astype(np.float32))   # (C,R,250)
    templates = np.einsum("nr,crt->cnt", A, source_time)              # (C,63,250)
    # Normalize each template to unit RMS so noise level is interpretable.
    templates /= (templates.std(axis=(1, 2), keepdims=True) + 1e-6)

    prototypes = rng.normal(0, 1, size=(n_concepts, CLIP_DIM)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9
    return templates.astype(np.float32), prototypes.astype(np.float32)


def make_dataset(n_concepts: int = 40, trials_per_concept: int = 40,
                 noise: float = 0.8, n_sources: int = 8, seed: int = 0,
                 task_scale: float = 1.0, distractor_scale: float = 0.0,
                 distractor_sources: int = 12):
    """Build a synthetic EEG dataset with per-channel z-normalization.

    Returns a dict with train/test tensors (CPU), concept prototypes, and the
    per-sample target prototype. Shapes: eeg (N,63,250), labels (N,), targets
    (N,512).

    When ``distractor_scale > 0`` each trial also gets a high-variance,
    concept-INDEPENDENT background (its own spatial mixing + random temporal
    coefficients). With ``task_scale`` small relative to ``distractor_scale``
    the discriminative signal is a *low-variance* part of the signal, so an
    MSE-only codec at low rate spends its budget on the background and drops
    the concept signature -- the regime where task-aware training matters.
    """
    templates, prototypes = _concept_bank(n_concepts, n_sources, seed)
    rng = np.random.default_rng(seed + 1)
    N = n_concepts * trials_per_concept
    labels = np.repeat(np.arange(n_concepts), trials_per_concept)
    eeg = task_scale * templates[labels]
    if distractor_scale > 0:
        t = np.arange(N_T)
        lat = [20, 40, 70, 110, 150, 190, 220]
        wid = [10, 12, 15, 18, 20, 16, 24]
        Bd = np.stack([np.exp(-0.5 * ((t - l) / w) ** 2) for l, w in zip(lat, wid)])
        Bd = np.concatenate([Bd, np.stack([np.sin(2 * np.pi * f * t / 250.0)
                                           for f in (4.0, 8.0, 12.0)])], axis=0)
        Ad = rng.normal(0, 1, size=(N_CH, distractor_sources)).astype(np.float32)
        coeff = rng.normal(0, 1, size=(N, distractor_sources, Bd.shape[0])).astype(np.float32)
        src_t = np.einsum("nsk,kt->nst", coeff, Bd.astype(np.float32))
        bg = np.einsum("cs,nst->nct", Ad, src_t)                     # (N,63,250)
        bg /= (bg.std(axis=(1, 2), keepdims=True) + 1e-6)            # unit RMS/trial
        eeg = eeg + distractor_scale * bg
    eeg = eeg + noise * rng.normal(0, 1, size=(N, N_CH, N_T)).astype(np.float32)
    eeg = eeg.astype(np.float32)

    # Per-channel z-normalization (mean/std over the whole set), like data.py.
    mean = eeg.mean(axis=(0, 2), keepdims=True)
    std = eeg.std(axis=(0, 2), keepdims=True) + 1e-6
    eeg = (eeg - mean) / std

    # Shuffle then split 80/20.
    perm = rng.permutation(N)
    eeg, labels = eeg[perm], labels[perm]
    n_test = N // 5
    sl_test = slice(0, n_test)
    sl_train = slice(n_test, N)

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    proto_t = to_t(prototypes)
    return {
        "eeg_train": to_t(eeg[sl_train]),
        "eeg_test": to_t(eeg[sl_test]),
        "lab_train": torch.from_numpy(labels[sl_train]).long(),
        "lab_test": torch.from_numpy(labels[sl_test]).long(),
        "prototypes": proto_t,                       # (C,512) unit-norm
        "tgt_train": proto_t[torch.from_numpy(labels[sl_train]).long()],
        "tgt_test": proto_t[torch.from_numpy(labels[sl_test]).long()],
        "n_concepts": n_concepts,
    }


# --------------------------------------------------------------------------- #
# Tiny codec / projector factories (light on VRAM)
# --------------------------------------------------------------------------- #

TINY = dict(hidden=32, c_lat=16, n_attn=0)   # latent (16,32)=512 symbols


def make_codec(hidden: int = 32, c_lat: int = 16, n_attn: int = 0):
    from codec import EEGCodec
    return EEGCodec(c_lat=c_lat, hidden=hidden, n_attn=n_attn)


def make_projector(hidden: int = 32, n_attn: int = 0):
    from clip_proj import EEGProjector
    return EEGProjector(hidden=hidden, n_attn=n_attn)


# --------------------------------------------------------------------------- #
# Shared training / eval utilities
# --------------------------------------------------------------------------- #

def iterate(*tensors, batch: int, shuffle: bool, dev: str):
    n = tensors[0].shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch):
        j = idx[i:i + batch]
        yield tuple(t[j].to(dev) for t in tensors)


def latent_bpp(bits_per_symbol: float, c_lat: int, t_lat: int = 32) -> float:
    """Bits per real EEG sample (matches codec.bpp_floor)."""
    return bits_per_symbol * (c_lat * t_lat) / (N_CH * N_T)


def ratio_vs_fp16(bpp: float) -> float:
    return 16.0 / max(bpp, 1e-9)


@torch.no_grad()
def collect_latents(codec, eeg, dev: str, batch: int = 128) -> torch.Tensor:
    """Integer-rounded latents for the whole set (inference path)."""
    codec.eval()
    out = []
    for (x,) in iterate(eeg, batch=batch, shuffle=False, dev=dev):
        y = codec.encoder(x)
        out.append(torch.round(y).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def retrieval_topk(query_emb: torch.Tensor, gallery: torch.Tensor,
                   gold_idx: torch.Tensor, ks=(1, 5, 10)) -> dict:
    import torch.nn.functional as F
    q = F.normalize(query_emb, dim=-1)
    g = F.normalize(gallery, dim=-1)
    sims = q @ g.t()
    ranks = sims.argsort(dim=-1, descending=True)
    out = {}
    for k in ks:
        hit = (ranks[:, :k] == gold_idx.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"top{k}"] = round(hit, 4)
    return out


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.dt = time.time() - self.t


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
