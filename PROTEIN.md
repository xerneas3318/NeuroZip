# Protein codec — the v4 architecture, a new domain

Branch `protein` (off `rian`). Demonstrates the NeuroZip v4 codec is
**domain-agnostic**: the exact same conv encoder/decoder + factorized-Laplace
prior + noise/round quantizer that compresses EEG also compresses **protein
sequences** — with *no structural change*, only the input channel count.

## The analogy (literally the same tensor shape)

| | EEG (NeuroZip) | Protein (this branch) |
|---|---|---|
| input | `(63 channels × 250 time)` | `(20 amino acids × 250 residues)` |
| representation | normalized μV | one-hot over the 20 AAs |
| codec | `EEGCodec` (v4, n_attn=0) | `ProteinCodec` (v4, n_attn=0) |
| latent | `(c_lat × 32)` scalar + Laplace | **identical** |
| rate model | factorized Laplace | **identical** |
| recon loss | MSE (continuous) | cross-entropy (categorical) |

`protein_codec.py` *reuses rian's exact building blocks* (`ConvNormAct`,
`FactorizedLaplacePrior`, `quantize`, `TransformerStack`) — the only difference
between `ProteinEncoder/Decoder` and `EEGEncoder/Decoder` is `63 → 20` in the
first conv and last conv. Everything else is byte-for-byte the same.

## Dataset

UniProt **SwissProt** (reviewed sequences, `data_protein/uniprot_sprot.fasta.gz`).
`protein_data.py` parses the FASTA, keeps the 196k+4k sequences of length 40–250
with only standard amino acids, and encodes each to a `(20 × 250)` one-hot
("image"), padding/cropping to 250 residues.

## Result (held-out, no VQ/RQ — pure v4 scalar codec)

Near-lossless reconstruction of real protein sequences:

| metric | value |
|---|---|
| per-residue reconstruction accuracy | **99.5%** (chance = 5%) |
| size vs **FASTA** (1 byte/residue, the real storage) | **smaller** — e.g. 237 B → 181 B (1.3×) |

> Honest note on size: the one-hot input is a strawman (it's deliberately
> redundant). Against the way proteins are actually stored — **FASTA, 1 byte per
> residue** — the codec is genuinely smaller at ~99% accuracy. It does **not**
> beat the information-theoretic floor (~4.32 bits/residue "packed"), and nothing
> high-fidelity can: protein sequences are near-maximum-entropy (~4.2 bits/res),
> so you only beat packed by sacrificing accuracy (c_lat=4 → 3.8 bits/res but 47%).
> The viewer reports FASTA, packed, and the compressed size side by side.

Reconstructed sequences are character-for-character matches
(`results/protein_reconstruction.png`):

```
orig:  MKTILPAVLFAAFATTSAWAAESVQPLEKIAPYPQAEKGMKRQVIQLTPQEDESTLKVELLIGQTLEVDC...  (len 162)
recon: MKTILPAVLFAAFATTSAWAAESVQPLEKIAPYPQAEKGMKRQVIQLTPQEDESTLKVELLIGQTLEVDC...  100.0%
```

The same rate knob (`--lambda-rate`) trades accuracy for bits, exactly as on EEG
(`results/protein_rate_distortion.png`): λ 0.02→0.5 stays ≥99.5% at 47–71×.

## Run

```bash
python protein_data.py                                   # download/parse + summary
python train_protein.py --epochs 25 --lambda-rate 0.5    # v4 codec on proteins
python make_protein_results.py                           # reconstruction demo figure
```

## Takeaway

One architecture, two unrelated modalities (brain signals and protein sequences),
no redesign — just point it at a `(channels × length)` tensor. The NeuroZip v4
codec is a general 1-D sequence compressor.
