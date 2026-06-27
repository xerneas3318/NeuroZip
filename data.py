"""NeuroZip data loading.

THINGS-EEG layout (per subject, as packaged by Haitao999/things-eeg):
  Preprocessed_data_250Hz_whiten/sub-XX/{train,test}.pt
    dict with:
      eeg     (N, R, 63, 250)  float16   -- N image-trials, R repetitions
      label   (N, R)            int64
      img     (N, R)            string    -- relative image path
      text    (N, R)            string    -- clean concept name (no prefix)
      ch_names list[str]        63 channels
  Preprocessed_data_250Hz_whiten/ViT-B-32_features_{train,test}.pt
    dict with:
      text_features {concept -> tensor(512)}
      img_features  {rel_path -> tensor(512)}

Train: N=16540, R=4   -> 66,160 epochs
Test:  N=200,   R=80  -> 16,000 epochs

Each row of eeg shares the same image+concept across R repetitions.

NORMALIZATION (one of the 3 easy-to-get-wrong spots in the project):
  Per-channel mean/std on the TRAIN epochs. Apply the same stats to test.
  We save the stats to data/norm_stats.pt so we can (a) invert the codec
  output for real-microvolt reconstruction, and (b) report a real bpp
  numerator (the bits we'd actually need to ship are bits-per-normalized-
  sample; without the stats you have no ground truth to compare against).
"""

from __future__ import annotations
import os, json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = Path(__file__).parent / "data"
EEG_DIR = DATA_ROOT / "Preprocessed_data_250Hz_whiten"
IMG_DIR = DATA_ROOT / "Image_set"
CLIP_DIM = 512                       # ViT-B-32
N_CHANNELS = 63
N_TIMES = 250
SR = 250                             # Hz (THINGS-EEG resampled to 250 Hz)


def _load_split(subject: str, split: str):
    p = EEG_DIR / subject / f"{split}.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run scripts/download_data.sh first.")
    return torch.load(p, weights_only=False, map_location="cpu")


def _load_clip_features(split: str) -> dict:
    p = EEG_DIR / f"ViT-B-32_features_{split}.pt"
    return torch.load(p, weights_only=False, map_location="cpu")


def _normalize_text_key(s: str) -> str:
    # CLIP feature dicts use the bare concept ("aardvark"), text col already clean.
    return str(s)


def compute_or_load_norm_stats(subject: str = "sub-01") -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std over the train split. Save once, reuse."""
    cache = DATA_ROOT / "norm_stats.pt"
    if cache.exists():
        s = torch.load(cache, weights_only=False, map_location="cpu")
        return s["mean"], s["std"]
    d = _load_split(subject, "train")
    eeg = torch.from_numpy(d["eeg"]).float()        # (N, R, 63, 250)
    flat = eeg.permute(2, 0, 1, 3).reshape(N_CHANNELS, -1)
    mean = flat.mean(dim=1)
    std = flat.std(dim=1).clamp_min(1e-6)
    torch.save({"mean": mean, "std": std}, cache)
    print(f"[norm] computed channel stats (per-channel mean range "
          f"{mean.min():.3f}..{mean.max():.3f}, std range "
          f"{std.min():.3f}..{std.max():.3f}) -> {cache}")
    return mean, std


class ThingsEEG(Dataset):
    """One sample = one (single-trial) EEG epoch + its image's CLIP feature + concept text."""

    def __init__(self, split: str = "train", subject: str = "sub-01",
                 normalize: bool = True, max_epochs: Optional[int] = None):
        assert split in ("train", "test")
        self.split = split
        d = _load_split(subject, split)
        feat = _load_clip_features(split)
        self.img_feats: dict = feat["img_features"]
        self.text_feats: dict = feat["text_features"]

        eeg = d["eeg"].astype(np.float32)            # (N, R, 63, 250)
        img = d["img"]                                # (N, R) str
        text = d["text"]                              # (N, R) str
        # Each (n,r) is an epoch. Flatten to a list.
        N, R = eeg.shape[:2]
        self.eeg = eeg.reshape(N * R, N_CHANNELS, N_TIMES)
        # We also keep (N, R) index for trial averaging at eval.
        self.n_concepts = N
        self.n_reps = R
        self.img_paths = img.reshape(N * R)
        self.texts = text.reshape(N * R)

        if max_epochs is not None:
            sel = np.random.default_rng(0).choice(len(self.eeg), size=max_epochs, replace=False)
            sel.sort()
            self.eeg = self.eeg[sel]
            self.img_paths = self.img_paths[sel]
            self.texts = self.texts[sel]

        # Pre-stack the per-epoch CLIP image feature for fast batching.
        clip_dim = next(iter(self.img_feats.values())).shape[-1]
        feats = np.empty((len(self.eeg), clip_dim), dtype=np.float32)
        missing = 0
        for i, p in enumerate(self.img_paths):
            v = self.img_feats.get(str(p))
            if v is None:
                missing += 1
                feats[i] = 0
            else:
                feats[i] = v.cpu().numpy()
        if missing:
            print(f"[warn] {missing}/{len(self.eeg)} epochs had no CLIP image feature")
        self.clip_img = feats

        if normalize:
            mean, std = compute_or_load_norm_stats(subject)
            self.eeg = (self.eeg - mean.numpy()[None, :, None]) / std.numpy()[None, :, None]

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.eeg[i]),           # (63, 250)
            torch.from_numpy(self.clip_img[i]),      # (512,)
            str(self.texts[i]),                      # concept
            str(self.img_paths[i]),                  # image path
        )

    # --- trial-averaged eval helpers ---
    def trial_averaged(self) -> tuple[torch.Tensor, list[str], list[str]]:
        """Average the R repetitions of each concept -> (N, 63, 250)."""
        eeg = self.eeg.reshape(self.n_concepts, self.n_reps, N_CHANNELS, N_TIMES).mean(axis=1)
        # Concept and representative image (first rep)
        text = self.texts.reshape(self.n_concepts, self.n_reps)[:, 0].tolist()
        img = self.img_paths.reshape(self.n_concepts, self.n_reps)[:, 0].tolist()
        return torch.from_numpy(eeg), text, img

    def concept_clip_img(self) -> torch.Tensor:
        """One CLIP image feature per concept (using the representative image)."""
        clip = self.clip_img.reshape(self.n_concepts, self.n_reps, -1)[:, 0]
        return torch.from_numpy(clip)

    def concept_clip_text(self) -> tuple[torch.Tensor, list[str]]:
        """CLIP text feature per concept (cached from feature file)."""
        concepts = []
        seen = set()
        for t in self.texts:
            if t not in seen:
                concepts.append(str(t)); seen.add(t)
        feats = []
        for c in concepts:
            v = self.text_feats.get(c)
            if v is None:
                v = torch.zeros(CLIP_DIM)
            feats.append(v)
        return torch.stack([f.float() for f in feats]), concepts


def make_loader(split: str = "train", batch_size: int = 256,
                shuffle: Optional[bool] = None, max_epochs: Optional[int] = None,
                num_workers: int = 2):
    ds = ThingsEEG(split=split, max_epochs=max_epochs)
    if shuffle is None:
        shuffle = (split == "train")
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=(split == "train"))


def summarize():
    """Stage-0 deliverable: print a summary of the loaded data."""
    print("\n=== NeuroZip data summary ===")
    train = ThingsEEG(split="train")
    test = ThingsEEG(split="test")
    print(f"train epochs: {len(train):>7}   concepts: {train.n_concepts}   reps: {train.n_reps}")
    print(f"test  epochs: {len(test):>7}   concepts: {test.n_concepts}   reps: {test.n_reps}")
    print(f"EEG shape per epoch: {train[0][0].shape}  dtype: {train[0][0].dtype}")
    e = train[0][0]
    print(f"normalized EEG range: min={e.min():.3f}  max={e.max():.3f}  "
          f"mean={e.mean():.4f}  std={e.std():.4f}")
    print(f"CLIP image feature dim: {train[0][1].shape}")
    print(f"distinct train concepts: {len(set(train.texts.tolist()))}")
    print(f"distinct test concepts:  {len(set(test.texts.tolist()))}")
    print(f"channels: {N_CHANNELS}  timepoints: {N_TIMES}  sample rate: {SR} Hz")
    mean, std = compute_or_load_norm_stats()
    print(f"per-channel mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"per-channel std  range: [{std.min():.3f}, {std.max():.3f}]")
    print("=============================\n")


if __name__ == "__main__":
    summarize()
