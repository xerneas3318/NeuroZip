"""
fable_io.py -- real compress -> bytes -> decompress for the fMRI VQ codec.

The codec turns each volume into a grid of P integer tokens. Successive fMRI
volumes barely change (measured: 44% of tokens repeat frame-to-frame, and
H(token_t | token_{t-1}) = 2.9 bits vs 7.8 marginal). So the compression comes
from **range-coding the tokens across time** with a first-order temporal model:
each token is coded under P(cur | prev-token-at-same-position), learned once on
the training tokens and shipped with the codec.

The .nzf artifact is SELF-CONTAINED: it stores the entropy-coded tokens plus the
tiny static side-info (brain mask, per-voxel baseline mu, global scale) needed to
put the signal back in original units with no external state. We report both the
self-contained size and the token-stream-only size (the marginal cost of one more
session given a per-subject codec).
"""

from __future__ import annotations
import struct
import numpy as np
import constriction

MAGIC = b"NZF1"
EPS = 1e-6


# ---------- temporal prior (fit on training tokens, shipped with codec) ----------

def fit_prior(codes_2d, K, beta=0.2):
    """codes_2d: (T, P) integer tokens in temporal order.
    Returns marginal (K,) and transition (K, K) = P(cur | prev), backed off to
    the marginal so every entry is > 0 (range coder needs full support)."""
    flat = codes_2d.reshape(-1)
    marg = np.bincount(flat, minlength=K).astype(np.float64) + EPS
    marg /= marg.sum()
    prev = codes_2d[:-1].reshape(-1)
    cur = codes_2d[1:].reshape(-1)
    trans = np.zeros((K, K), np.float64)
    np.add.at(trans, (prev, cur), 1.0)
    trans += beta * marg[None, :]            # back-off to marginal for rare contexts
    trans += EPS
    trans /= trans.sum(1, keepdims=True)
    return {"marginal": marg, "transition": trans, "K": int(K)}


def _probs_for_stream(codes_2d, prior):
    """Per-symbol distributions for the whole (T,P) token stream, in row-major
    (frame, position) order: frame 0 uses the marginal, frame t uses
    transition[prev-token]."""
    T, P = codes_2d.shape
    K = prior["K"]
    probs = np.empty((T, P, K), np.float64)
    probs[0] = prior["marginal"][None, :]
    if T > 1:
        probs[1:] = prior["transition"][codes_2d[:-1]]
    return probs.reshape(T * P, K)


def encode_tokens(codes_2d, prior):
    probs = _probs_for_stream(codes_2d, prior)
    model = constriction.stream.model.Categorical(perfect=False)
    enc = constriction.stream.queue.RangeEncoder()
    enc.encode(codes_2d.reshape(-1).astype(np.int32), model, probs)
    return enc.get_compressed().tobytes()


def decode_tokens(blob, T, P, prior):
    K = prior["K"]
    comp = np.frombuffer(blob, dtype=np.uint32).copy()
    dec = constriction.stream.queue.RangeDecoder(comp)
    model = constriction.stream.model.Categorical(perfect=False)
    codes = np.empty((T, P), np.int32)
    # sequential: frame t's distribution depends on frame t-1 (already decoded)
    codes[0] = dec.decode(model, np.repeat(prior["marginal"][None, :], P, axis=0))
    for t in range(1, T):
        codes[t] = dec.decode(model, prior["transition"][codes[t - 1]])
    return codes


# ---------- mask bit-packing ----------

def _pack_mask(mask):
    return np.packbits(mask.reshape(-1).astype(np.uint8)).tobytes()


def _unpack_mask(blob, shape):
    n = int(np.prod(shape))
    bits = np.unpackbits(np.frombuffer(blob, np.uint8), count=n)
    return bits.reshape(shape).astype(bool)


# ---------- container ----------

def compress(vols, codec, prior, mu, gstd, mask, device="cuda"):
    """vols: (T, D, H, W) float32, ALREADY normalized ((x-mu)/gstd, background 0),
    in temporal order. Returns (blob_bytes, info)."""
    import torch
    codec.eval()
    D, H, W = mask.shape
    with torch.no_grad():
        x = torch.from_numpy(vols).unsqueeze(1).to(device)
        codes = torch.cat([codec.encode(x[i:i + 128]) for i in range(0, len(x), 128)])
    codes_2d = codes.reshape(len(vols), -1).cpu().numpy().astype(np.int32)
    T, P = codes_2d.shape

    tok_blob = encode_tokens(codes_2d, prior)
    mask_blob = _pack_mask(mask)
    mu16 = mu.astype(np.float16).tobytes()          # per-voxel baseline (in-brain)

    header = MAGIC + struct.pack(
        "<IHHHHHHHfII",
        T, *codec.grid, prior["K"], D, H, W, float(gstd),
        len(mask_blob), len(mu16))
    blob = header + mask_blob + mu16 + struct.pack("<I", len(tok_blob)) + tok_blob
    info = dict(n_vols=T, tokens_per_vol=P, token_bytes=len(tok_blob),
                mask_bytes=len(mask_blob), mu_bytes=len(mu16),
                total_bytes=len(blob))
    return blob, info


def decompress(blob, codec, device="cuda"):
    """Returns (vols_normalized (T,D,H,W), mu, gstd, mask)."""
    import torch
    off = len(MAGIC)
    (T, gd, gh, gw, K, D, H, W, gstd, n_mask, n_mu) = struct.unpack_from("<IHHHHHHHfII", blob, off)
    off += struct.calcsize("<IHHHHHHHfII")
    mask = _unpack_mask(blob[off:off + n_mask], (D, H, W)); off += n_mask
    mu = np.frombuffer(blob[off:off + n_mu], np.float16).astype(np.float32); off += n_mu
    (n_tok,) = struct.unpack_from("<I", blob, off); off += 4
    tok_blob = blob[off:off + n_tok]

    P = gd * gh * gw
    # prior is part of the codec bundle; attached by caller as codec._prior
    codes_2d = decode_tokens(tok_blob, T, P, codec._prior)
    codes = torch.from_numpy(codes_2d.reshape(T, gd, gh, gw)).long().to(device)
    codec.eval()
    with torch.no_grad():
        vols = torch.cat([codec.decode(codes[i:i + 128]) for i in range(0, T, 128)])
    vols = vols.squeeze(1).cpu().numpy()
    vols[:, ~mask] = 0.0
    return vols, mu, gstd, mask
