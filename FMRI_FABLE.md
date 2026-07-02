# fMRI-fable — compressing the 4D volume, not the ROI time series

Branch `fmri-fable`. The brief: get fMRI compression *and decompression* actually
working, with a **small** model, using **VQ** or whatever fits, on a dataset I
pick — the earlier fMRI branches "didn't work great."

## Why the earlier branches stalled — and the fix

- The `fmri` branch compressed **parcellated CC200 ROI time series** (200×T) and
  honestly concluded the signal is *high-rank* (68% variance at 83×). That is
  true — but it is a consequence of the representation: parcellation has
  **already removed the spatial redundancy**, so what's left is nearly
  incompressible. It also **z-scored per ROI**, which whitens away what little
  structure remains.
- The thing that actually costs disk is the **4D volume** (X·Y·Z·T), and a brain
  volume is enormously redundant: ~75% is background, in-brain signal is spatially
  smooth, and **successive TRs are ~identical** (measured successive-volume spatial
  correlation 0.96 on this data). That redundancy is exactly what a small codec
  can exploit. So this branch compresses the **volume**.

Two normalization facts, verified by diagnostics rather than guessed:
- **Do not z-score per voxel.** It forces every voxel to unit variance,
  amplifying noise voxels and destroying the low-rank spatial structure. With one
  *global* scale instead, the per-volume signal is genuinely low-rank: rank-16 PCA
  already explains 74% of held-out variance, rank-64 → 83%.
- Store the per-voxel **mean image** (baseline / anatomy) once as tiny side-info
  and compress only the **temporal fluctuation** (the functional signal). All
  fidelity numbers below are on that fluctuation — the honest, hard part.

## Dataset

Haxby 2001 (`nilearn.datasets.fetch_haxby`, subject 1): a real 4D task-fMRI run,
(40,64,64,1452) int16, 12 runs, 9 stimulus classes. Cropped to the brain box
(40,56,48). **Split by run**: runs 0–9 train, runs 10–11 held out — the codec is
scored on volumes it never saw. `fable_data.py`.

## Architecture — small 3D VQ-VAE + temporal token coding

`fable_codec.py` (≤2.4M params):

```
volume (1,40,56,48) --3D conv encoder (n_down stride-2 stages)--> z_e (D,gd,gh,gw)
     --VectorQuantizer (EMA codebook, dead-code restart)--> P integer tokens + z_q
     --3D conv decoder (mirror)--> volume_hat
```

- **VQ** (not scalar+Laplace): the RQ-VAE branch showed the two *tie* on fidelity,
  but VQ gives a clean fixed-length token grid per volume — which is what makes the
  temporal bitstream below simple. EMA codebook with a **dead-code restart** keeps
  all K codes live (the RQ-VAE branch collapsed to 6/1024 without it; here 495–512
  of 512 stay used).
- **LayerScale is load-bearing.** Plain pre-activation residual blocks made the
  whole 3D AE collapse to predicting the per-voxel mean (MSE = signal variance,
  vanishing gradients — verified). Initialising each residual branch with a small
  learnable scale (γ=1e-2, near-identity) fixes it. Documented in the code.
- **Rate = range-coding the tokens across time.** `fable_io.py` fits a first-order
  temporal model P(token_t | token_{t-1} at same position) on the training tokens
  and range-codes with it (constriction). 44% of tokens simply repeat frame to
  frame; H(token_t | token_{t-1}) = **2.9 bits** vs 7.8 marginal — a 3× rate cut
  that is where the compression actually comes from. The entropy layer is exactly
  lossless (verified bit-exact round-trip).

The `.nzf` artifact is self-contained: entropy-coded tokens + brain mask +
baseline μ + global scale, enough to reconstruct original scanner units with no
external state. `fable_cli.py` compresses/decompresses ordinary NIfTI files.

## Results (held-out runs 10–11, 242 volumes the codec never saw)

`fable_report.py`; ratio is real bytes of the self-contained `.nzf` vs the
gzipped `.nii.gz`; VE and corr are on the temporal fluctuation (in-brain); VT
decode is 8-way object decoding through ventral-temporal cortex (chance 12.5%).

| codec | params | tokens/vol | ratio | B/vol | VE | corr | PSNR | VT decode retained |
|-------|-------:|-----------:|------:|------:|----:|-----:|-----:|-------------------:|
| `nd3` | 2.40M  |   210      | **294×** |  516 | 71.8% | 0.358 | 38.2 dB | 29% (16.7/58.3) |
| `nd2` | 1.43M  |  1680      |  116× | 1306 | 67.9% | 0.309 | 37.6 dB | 42% (24.3/58.3) |
| `nd1` | 0.59M  | 13440      |  **26×** | 5755 | 82.4% | 0.610 | 40.2 dB | **100% (58.3/58.3)** |

Reading the curve (`plots/fable_rd.png`):

- **Storage/fidelity works great.** The `nd3` codec compresses the held-out
  session **294× vs the gzipped .nii.gz a lab actually stores** at 71.8% VE / 38 dB
  on the fluctuation (the full-signal round-trip, baseline + fluctuation, is
  ~43 dB). A real archival/transfer codec — and via `fable_cli.py` the entire
  308 MB Haxby scan (all 12 runs) compresses to **232 KB**.
- **Fidelity saturates; *function* is what costs bits — and it's preserved.**
  Variance-explained flattens near the signal's own ceiling (per-volume linear PCA
  tops out ~88%). The interesting axis is decodability: at 294× the fine VT
  pattern is gone (all 577 VT voxels fall under ~1 latent cell), but the
  function-preserving `nd1` keeps **100%** of within-representation decoding and
  **98%** cross-representation (a decoder trained on *original* data reads the
  decompressed brains at 56.9% vs 58.3% baseline) — at 26× and, notably, the
  **smallest** model (0.59M). The rate–function plot is monotonic: 29 → 42 → 100%.

So the per-volume codec gives two operating points from one small architecture: a
~300× archival codec and a ~26× codec that preserves the neuroscience (`nd1` keeps
100% of decoding). This is the NeuroZip thesis re-derived from the compression
side — MSE keeps the high-variance signal and drops the low-variance task signal.

## Getting MSE well under 0.1 — compress the TIME axis (`fable_svd.py`)

The per-volume codec's reconstruction MSE floors around **0.24** no matter how
it's tuned, and diagnostics say why: it compresses the wrong axis. A *single* fMRI
volume is high-rank in space (per-volume PCA floors at MSE ~0.14 even at rank
1000), so compressing each volume independently is intrinsically hard. But an fMRI
*run* is dramatically low-rank in **time** — it's a block design and BOLD is slow,
so a rank-16 temporal SVD of the 121-TR run reconstructs to **MSE 0.056**. The
per-volume codec throws that away (it only used time to entropy-code tokens).

`fable_svd.py` compresses along the redundant axis. Per temporal segment it takes
an SVD of the (T×voxels) matrix, keeps the top-K components, and stores K
**full-resolution** spatial maps (int8) + tiny per-frame temporal coefficients
(float16), gzipped. Full spatial resolution is retained (that's what drives MSE
down); only time is truncated. It's classic transform (KLT) coding, no training —
the basis is computed per segment at compress time. Real end-to-end bytes:

| rank K | MSE | VE | PSNR | ratio vs .nii.gz | VT decode retained |
|-------:|----:|---:|-----:|-----------------:|-------------------:|
| 8  | **0.089** | 93.5% | 44.6 dB | **75×** | 35% |
| 12 | 0.071 | 94.8% | 45.6 dB | 52× | 35% |
| 16 | 0.059 | 95.7% | 46.3 dB | 41× | 33% |

Rank-8 already clears the target: **MSE 0.089 at 75×** — lower MSE *and* higher
ratio than any per-volume setting, because it finally exploits the temporal
redundancy. (A neural spatiotemporal conv codec was also tried; it stays
spatial-grid-limited — a downsampled latent can't hold full spatial detail — which
is exactly why the transform codec, keeping full-resolution maps, wins here.)

The catch is the mirror image of `nd3`: truncating to the top-K *variance*
directions keeps the dominant signal (low MSE) but drops the low-variance VT
decoding pattern (35% retained). So the two codec families sit at opposite corners
of the fidelity–function trade:

- **`fable_svd.py`** — lowest MSE (0.06–0.09), highest ratio (40–75×), fine
  decoding lost. Best for **archival / fidelity**.
- **`fable_codec.py` `nd1`** — higher MSE (0.24) but 100% decoding retained. Best
  when the **downstream analysis** must survive compression.

Pick by what the compressed data is for. Both are small; the SVD codec needs no
trained weights at all.

## Test it (one command)

The trained checkpoints are committed, so after fetching the data you can verify
every codec end-to-end (metrics table + a real compress→file→decompress round-trip):

```bash
pip install -r requirements.txt          # + nibabel nilearn scipy scikit-learn constriction
python fable_data.py                      # one-time: fetch Haxby (~300 MB) + build cache
python fable_test.py                      # runs nd1 / nd3 / svd-8 / svd-16, prints the table
```

Expected output (held-out session):

```
codec            params     MSE     VE    PSNR   ratio   B/vol  decode kept
nd1 neural        0.59M   0.241    82%   40.2dB     26x    5755        100%
nd3 neural        2.40M   0.384    72%   38.2dB    294x     516         29%
svd K=8      0 (no net)   0.089    93%   44.6dB     75x    2014         35%
svd K=16     0 (no net)   0.059    96%   46.3dB     41x    3697         33%
```

## Run (individual tools)

```bash
python fable_train.py --n-down 3 --K 512 --epochs 150 --out checkpoints/fable_nd3.pt
python fable_finalize.py --ckpt checkpoints/fable_nd3.pt   # bundle temporal prior
python fable_eval.py   --ckpt checkpoints/fable_nd3.pt     # rate / fidelity / decoding
python fable_report.py checkpoints/fable_nd1.pt checkpoints/fable_nd2.pt checkpoints/fable_nd3.pt

# low-MSE temporal-SVD codec (no training needed)
python fable_svd.py --rank 8              # MSE 0.089 @ 75x
python fable_svd.py --rank 16             # MSE 0.059 @ 41x

# real files (per-volume neural codec)
python fable_cli.py compress   data_fmri/haxby2001/subj1/bold.nii.gz -o out.nzf --ckpt checkpoints/fable_nd3.pt
python fable_cli.py decompress out.nzf -o recon.nii.gz --ckpt checkpoints/fable_nd3.pt
```
