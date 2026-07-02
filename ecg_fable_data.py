"""
ecg_fable_data.py -- PTB-XL 12-lead ECG (100 Hz) WITH diagnostic labels.

The existing `ecg` branch compresses ECG windows and reports reconstruction only.
For a paper we need the NeuroZip *task-preservation* metric: does the clinical
diagnosis survive compression? PTB-XL ships a 5-way diagnostic superclass label
per record (NORM / MI / STTC / HYP / CD) and an official 10-fold split.

This loader builds, from the single-superclass records (clean labels):
  - record-level tensors (N, 12, 1000) + labels + folds  -> diagnosis eval
  - (12, 250) windows split by fold                       -> codec train/eval
Per-lead z-score (stats from train folds). Train = folds 1-8, val = 9, test = 10.
"""
from __future__ import annotations
import os, ast
import numpy as np
import pandas as pd

ROOT = "data_ecg"
CACHE = "data_ecg/ecg_fable_cache.npz"
N_LEADS, REC_LEN, WIN = 12, 1000, 250
SUPERCLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]


def _labeled_meta():
    df = pd.read_csv(f"{ROOT}/ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(f"{ROOT}/scp_statements.csv", index_col=0)
    diag = scp[scp.diagnostic == 1]

    def sup(codes):
        d = ast.literal_eval(codes)
        return sorted({diag.loc[c].diagnostic_class for c in d
                       if c in diag.index and pd.notna(diag.loc[c].diagnostic_class)})
    df["sc"] = df.scp_codes.apply(sup)
    df = df[df.sc.apply(len) == 1].copy()          # single-superclass = clean 5-way label
    df["label"] = df.sc.apply(lambda s: SUPERCLASSES.index(s[0]))
    return df


def build_cache(max_records=None, cache=CACHE):
    import wfdb
    df = _labeled_meta()
    if max_records:
        df = df.iloc[:max_records]
    recs, labels, folds = [], [], []
    for eid, row in df.iterrows():
        path = f"{ROOT}/{row.filename_lr}"
        if not os.path.exists(path + ".dat"):
            continue
        try:
            sig, _ = wfdb.rdsamp(path)             # (1000, 12)
        except Exception:
            continue
        recs.append(np.nan_to_num(sig.T.astype(np.float32)))
        labels.append(int(row.label)); folds.append(int(row.strat_fold))
    R = np.stack(recs)                             # (N, 12, 1000)
    labels = np.array(labels); folds = np.array(folds)
    tr = folds <= 8
    mean = R[tr].mean((0, 2), keepdims=True)
    std = R[tr].std((0, 2), keepdims=True) + 1e-6
    R = (R - mean) / std
    np.savez(cache, recs=R.astype(np.float32), labels=labels, folds=folds,
             mean=mean, std=std)
    print(f"[ecg] cache -> {cache}: {len(R)} labeled records, "
          f"train/val/test = {(folds<=8).sum()}/{(folds==9).sum()}/{(folds==10).sum()}")
    return cache


def load(rebuild=False):
    if rebuild or not os.path.exists(CACHE):
        build_cache()
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


def windows(recs, win=WIN):
    """(N,12,1000) -> (N*4,12,250) non-overlapping windows (codec epochs)."""
    N = len(recs)
    n = REC_LEN // win
    return recs[:, :, :n * win].reshape(N, N_LEADS, n, win).transpose(0, 2, 1, 3).reshape(N * n, N_LEADS, win)


if __name__ == "__main__":
    build_cache()
    d = load()
    print("=" * 56)
    print("PTB-XL 12-lead ECG -- fable cache")
    print("=" * 56)
    print(f"  records      : {d['recs'].shape}  (12 leads x 10 s @ 100 Hz)")
    print(f"  classes      : {SUPERCLASSES}")
    import numpy as np
    for i, s in enumerate(SUPERCLASSES):
        print(f"    {s:5s}: {(d['labels']==i).sum()}")
    print(f"  test records : {(d['folds']==10).sum()}  windows/rec: {REC_LEN//WIN}")
