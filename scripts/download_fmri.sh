#!/usr/bin/env bash
# THINGS-fMRI: single-trial IT responses to the 100 test images (+ROIs, noise ceilings).
# ~150 MB (the full 8740-image betas are 43 GB; we only need the test set in IT).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data_fmri && cd data_fmri
curl -sL "https://ndownloader.figshare.com/files/41039093" -o betas_csv_testset.zip
curl -sL "https://ndownloader.figshare.com/files/38517326" -o rois.zip
curl -sL "https://ndownloader.figshare.com/files/36682266" -o noise_ceilings.zip
for z in betas_csv_testset rois noise_ceilings; do unzip -oq "$z.zip"; done
echo "done. now: python fmri_neurozip.py --subject sub-01  (then sub-02, sub-03)"
