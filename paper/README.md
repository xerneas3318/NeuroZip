# Paper draft — NeuroZip

Assembles the four application branches (`ecg-fable`, `fmri-fable`, `har-fable`) and
the `redundancy-spectrum` analysis into one paper.

- `PAPER.md` — full readable draft (abstract → discussion) with all result tables.
- `paper.tex` — compilable LaTeX version (`pdflatex paper.tex`).
- `figures/` — `redundancy_spectrum.png` (Fig. 1), `fmri_rate_distortion.png` (Fig. 2),
  extracted from the `redundancy-spectrum` / `fmri-fable` branches.

Central thesis: **compressibility is redundancy, and where the redundancy lives
dictates the codec.** A small codec compresses biosignals 30–300× while preserving
the downstream task (diagnosis, decoding, activity, retrieval) at 98–100% of
baseline. Numbers come from the per-branch findings docs (`ECG_FABLE.md`,
`FMRI_FABLE.md`, `HAR_FABLE.md`, `REDUNDANCY_SPECTRUM.md`).
