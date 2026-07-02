"""
har_fable_data.py -- UCI HAR wearable-sensor windows WITH activity labels.

A non-neural modality, to show the codec is a generic (channels x time)
compressor: 9 inertial channels (body/total acc x/y/z + gyro x/y/z) x 128 samples
(2.56 s @ 50 Hz), 6 activities (walking, walking up/down, sitting, standing,
laying). Official train/test subject split. Per-channel z-score (train stats).
"""
from __future__ import annotations
import os
import numpy as np

ROOT = "data_har/UCI HAR Dataset"
CACHE = "data_har/har_fable_cache.npz"
CHANS = ["body_acc_x", "body_acc_y", "body_acc_z",
         "body_gyro_x", "body_gyro_y", "body_gyro_z",
         "total_acc_x", "total_acc_y", "total_acc_z"]
ACTS = ["walking", "walk_up", "walk_down", "sitting", "standing", "laying"]
N_CH, N_TIMES = 9, 128


def _load_split(split):
    sig = [np.loadtxt(f"{ROOT}/{split}/Inertial Signals/{c}_{split}.txt", dtype=np.float32)
           for c in CHANS]                       # each (N, 128)
    X = np.stack(sig, axis=1)                     # (N, 9, 128)
    y = np.loadtxt(f"{ROOT}/{split}/y_{split}.txt", dtype=np.int64) - 1
    return X, y


def build_cache(cache=CACHE):
    Xtr, ytr = _load_split("train")
    Xte, yte = _load_split("test")
    mean = Xtr.mean((0, 2), keepdims=True); std = Xtr.std((0, 2), keepdims=True) + 1e-6
    Xtr = (Xtr - mean) / std; Xte = (Xte - mean) / std
    np.savez(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, mean=mean, std=std)
    print(f"[har] cache -> {cache}: train {Xtr.shape} test {Xte.shape}")
    return cache


def load(rebuild=False):
    if rebuild or not os.path.exists(CACHE):
        build_cache()
    d = np.load(CACHE, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build_cache()
    d = load()
    print("=" * 50)
    print("UCI HAR -- fable cache")
    print(f"  windows : {N_CH} channels x {N_TIMES} samples")
    print(f"  train/test : {len(d['Xtr'])}/{len(d['Xte'])}")
    for i, a in enumerate(ACTS):
        print(f"    {a:9s}: train {(d['ytr']==i).sum()} test {(d['yte']==i).sum()}")
