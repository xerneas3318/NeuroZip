#!/usr/bin/env bash
# Download only the slice of THINGS-EEG that NeuroZip actually needs.
# Avoids the 33k-file naive `datasets.load_dataset` which takes hours.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"

$PY - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
repo = "Haitao999/things-eeg"
out = "data"
os.makedirs(out, exist_ok=True)
needed = [
    "Preprocessed_data_250Hz_whiten/sub-01/test.pt",
    "Preprocessed_data_250Hz_whiten/sub-01/train.pt",
    "Preprocessed_data_250Hz_whiten/ViT-B-32_features_train.pt",
    "Preprocessed_data_250Hz_whiten/ViT-B-32_features_test.pt",
    "README.md",
]
for f in needed:
    p = hf_hub_download(repo, f, repo_type="dataset", local_dir=out)
    print(f"  got {f} -> {os.path.getsize(p)/1e6:.1f} MB")

# 200 test images (small).
snapshot_download(repo, repo_type="dataset", local_dir=out,
                  allow_patterns=["Image_set/test_images/*/*"])
print("Done.")
PY
