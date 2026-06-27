#!/usr/bin/env bash
# End-to-end NeuroZip pipeline: data -> projector -> codec -> task loss -> eval -> demo assets.
# Each stage is independently runnable; see instructions.md (gitignored) for details.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[0/4] data sanity"
python -m neurozip.data            # Stage 0: load + print shape summary

echo "[1/4] train EEG->CLIP projector (frozen judge)"
python -m neurozip.train --stage projector

echo "[2/4] train fidelity-only codec (baseline)"
python -m neurozip.train --stage codec --lambda-task 0

echo "[3/4] train task-aware codec (sweep lambda_task)"
python -m neurozip.train --stage codec --lambda-task 1

echo "[4/4] evaluate + regenerate demo assets"
python -m neurozip.evaluate

echo "done -> open ui/demo.html"
