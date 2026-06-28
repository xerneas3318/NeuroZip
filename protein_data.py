"""
protein_data.py — load UniProt SwissProt sequences as fixed-length amino-acid
tensors, analogous to the THINGS-EEG loader.

A protein sequence -> integer indices (0..19 for the 20 standard amino acids),
cropped/padded to N_POS=250 positions (pad index = 20). One-hot is built on the
fly per batch: (B, 20, 250), the protein analog of the (B, 63, 250) EEG epoch.

Run standalone for a summary:
    python protein_data.py
"""

import gzip
import os
import numpy as np

from protein_codec import AA_ORDER, N_AA, N_POS

AA_TO_IDX = {a: i for i, a in enumerate(AA_ORDER)}
PAD = N_AA                                  # 20 = padding token (one-hot -> all zeros)
FASTA = "data_protein/uniprot_sprot.fasta.gz"
CACHE = "data_protein/protein_idx.npz"


def _iter_fasta(path):
    seq = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if seq:
                    yield "".join(seq)
                seq = []
            else:
                seq.append(line.strip())
        if seq:
            yield "".join(seq)


def build_cache(fasta=FASTA, cache=CACHE, min_len=40, max_len=N_POS, max_seqs=200000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    kept = 0
    for s in _iter_fasta(fasta):
        L = len(s)
        if L < min_len or L > max_len:
            continue
        if any(c not in AA_TO_IDX for c in s):      # skip non-standard (X,B,Z,U,O,*)
            continue
        idx = np.full(N_POS, PAD, dtype=np.int8)
        idx[:L] = [AA_TO_IDX[c] for c in s]
        rows.append((idx, L))
        kept += 1
        if kept >= max_seqs:
            break
    idxs = np.stack([r[0] for r in rows])           # (N, 250) int8
    lens = np.array([r[1] for r in rows], np.int16)  # (N,)
    perm = rng.permutation(len(idxs))
    idxs, lens = idxs[perm], lens[perm]
    np.savez(cache, idx=idxs, lens=lens)
    return idxs, lens


def load_proteins(val_frac=0.02, seed=0, rebuild=False):
    if rebuild or not os.path.exists(CACHE):
        idxs, lens = build_cache(seed=seed)
    else:
        d = np.load(CACHE); idxs, lens = d["idx"], d["lens"]
    n_val = int(len(idxs) * val_frac)
    val = (idxs[:n_val], lens[:n_val])
    train = (idxs[n_val:], lens[n_val:])
    meta = {"n_train": len(train[0]), "n_val": len(val[0]), "n_aa": N_AA, "n_pos": N_POS,
            "aa_order": AA_ORDER}
    return train, val, meta


def to_onehot(idx_batch, device):
    """idx_batch: (B, L) int -> (B, 20, L) float one-hot (pad positions all-zero)."""
    import torch
    idx = idx_batch.long().to(device)
    pad_mask = idx == PAD
    oh = torch.zeros(idx.shape[0], N_AA, idx.shape[1], device=device)
    safe = idx.clamp(max=N_AA - 1)
    oh.scatter_(1, safe.unsqueeze(1), 1.0)
    oh = oh * (~pad_mask).unsqueeze(1).float()       # zero out padding columns
    return oh


if __name__ == "__main__":
    train, val, meta = load_proteins()
    import collections
    idxs, lens = train
    print("=" * 56)
    print("UniProt SwissProt — protein sequence summary")
    print("=" * 56)
    print(f"  train / val sequences : {meta['n_train']} / {meta['n_val']}")
    print(f"  amino acids (channels): {meta['n_aa']}  ({meta['aa_order']})")
    print(f"  positions (length)    : {meta['n_pos']}  (crop/pad)")
    print(f"  length range          : {lens.min()}..{lens.max()}  median {int(np.median(lens))}")
    print(f"  one-hot epoch shape   : (20, 250)   <- analog of EEG (63, 250)")
    freq = collections.Counter(idxs[idxs != PAD].tolist())
    top = "".join(AA_ORDER[i] for i, _ in freq.most_common(5))
    print(f"  most frequent AAs     : {top}")
    print("=" * 56)
