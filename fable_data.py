"""
fable_data.py -- 4D fMRI volume loader for the NeuroZip `fmri-fable` codec.

The earlier `fmri` branch compressed *parcellated ROI time series* (200 x T).
That representation has already had its redundancy removed by the parcellation,
so it is high-rank and hard to compress (the branch's own honest conclusion).

This branch compresses the thing that actually takes up disk: the **4D volume**
itself (X x Y x Z x T). A brain volume is ~75% background and its in-brain
signal is spatially smooth and *temporally* almost static (successive-TR spatial
correlation ~0.96 on this data). That redundancy is exactly what a small codec
can exploit.

Representation choice (matters):
  raw signal  = per-voxel baseline mu(v)      (large, static -> stored once)
              + fluctuation  r_t(v)           (small, structured -> compressed)
So we z-score per voxel over the training TRs, keep (mu, sigma, mask) as tiny
static side-info, and hand the codec zero-background unit-variance volumes. The
codec then only has to model the functional fluctuation, not the anatomy.

Dataset: Haxby 2001 (nilearn `fetch_haxby`) -- a real 4D task-fMRI run,
(40,64,64,1452) int16, 12 runs, 9 stimulus classes. Split is by *run*: held-out
runs the codec never saw, so the eval is genuine generalization.
"""

from __future__ import annotations
import os
import numpy as np

DATA_DIR = os.environ.get("NZ_FMRI_DATA", "data_fmri")
CACHE = os.path.join(DATA_DIR, "fable_cache.npz")

# Fixed crop box that contains the Haxby brain mask, each dim divisible by 8
# (so 3 stride-2 downsamplings land on integers). Full volume is (40,64,64).
CROP = (slice(0, 40), slice(4, 60), slice(8, 56))     # -> (40, 56, 48)
VOL_SHAPE = (40, 56, 48)


def _load_haxby_subjects(n_subjects=1):
    """Download (once) and return list of (bold_4d, mask_3d, run_ids, class_ids)."""
    import nibabel as nib
    import pandas as pd
    from nilearn import datasets

    hx = datasets.fetch_haxby(subjects=list(range(1, n_subjects + 1)),
                              data_dir=DATA_DIR)
    mask = nib.load(hx.mask).get_fdata()[CROP] > 0            # (40,56,48) bool
    out = []
    for si in range(n_subjects):
        bold = nib.load(hx.func[si]).get_fdata(dtype=np.float32)     # (X,Y,Z,T)
        bold = bold[CROP[0], CROP[1], CROP[2], :]                    # crop
        lab = pd.read_csv(hx.session_target[si], sep=" ")
        runs = lab["chunks"].to_numpy()
        classes = lab["labels"].astype("category").cat.codes.to_numpy()
        class_names = list(lab["labels"].astype("category").cat.categories)
        out.append((bold, mask, runs, classes, class_names))
    return out


def build_cache(n_subjects=1, test_runs=(10, 11), cache=CACHE):
    """Crop, per-voxel z-score (stats from train runs only), split by run."""
    subs = _load_haxby_subjects(n_subjects)
    bold, mask, runs, classes, class_names = subs[0]        # single-subject codec
    T = bold.shape[-1]
    is_test = np.isin(runs, test_runs)

    vox = bold[mask]                                        # (Nvox, T)
    train_cols = ~is_test
    mu = vox[:, train_cols].mean(1)                        # per-voxel baseline (static side-info)
    # Scale by ONE global std, not per-voxel: per-voxel z-scoring whitens away
    # the low-rank spatial redundancy that makes fMRI compressible (it forces
    # every voxel to unit variance, amplifying noise voxels). A single global
    # scale keeps the natural variance structure intact.
    gstd = float((vox[:, train_cols] - mu[:, None]).std()) + 1e-6
    sd = np.full_like(mu, gstd)                            # per-voxel array kept for a uniform API
    z = (vox - mu[:, None]) / gstd                         # de-meaned, globally scaled fluctuation

    def to_vol(cols):
        v = np.zeros((int(cols.sum()), *VOL_SHAPE), np.float32)
        zc = z[:, cols].T                                  # (n, Nvox)
        v[:, mask] = zc
        return v

    train = to_vol(train_cols)
    test = to_vol(is_test)
    np.savez(cache,
             train=train, test=test,
             train_runs=runs[train_cols], test_runs=runs[is_test],
             train_cls=classes[train_cols], test_cls=classes[is_test],
             class_names=np.array(class_names),
             mask=mask, mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    print(f"[fable] cache -> {cache}")
    print(f"        train {train.shape}  test {test.shape}  (held-out runs {test_runs})")
    print(f"        in-brain voxels {int(mask.sum())} / {mask.size} "
          f"({100*mask.mean():.0f}% of crop)")
    return cache


def load(rebuild=False):
    if rebuild or not os.path.exists(CACHE):
        build_cache()
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    import sys
    build_cache(n_subjects=1, test_runs=(10, 11))
    d = load()
    print("=" * 60)
    print("Haxby 4D fMRI -- fable cache summary")
    print("=" * 60)
    tr, te = d["train"], d["test"]
    print(f"  volume shape      : {VOL_SHAPE}  (cropped from 40x64x64)")
    print(f"  train / test vols : {len(tr)} / {len(te)}")
    print(f"  test runs         : {sorted(set(d['test_runs'].tolist()))}")
    print(f"  classes           : {list(d['class_names'])}")
    print(f"  standardized range: [{tr.min():.1f}, {tr.max():.1f}]  std {tr[:, d['mask']].std():.2f}")
