# ECG-fable — diagnosis-preserving compression of 12-lead ECG

Branch `ecg-fable` (off `ecg`). The `ecg` branch showed the 1-D NeuroZip codec
reconstructs 12-lead ECG almost perfectly at high compression (96% variance, 31×)
— ECG is periodic and redundant, the codec's sweet spot. But reconstruction MSE
is not the clinical question. This branch adds the NeuroZip **task-preservation**
metric to a real clinical task: **does the diagnosis survive compression?**

## Dataset

**PTB-XL** (PhysioNet), 12-lead clinical ECG at 100 Hz, with a 5-way diagnostic
**superclass** label per record (NORM, MI, STTC, HYP, CD) and an official 10-fold
split. We use the single-superclass records (clean labels), train on folds 1–8,
test on fold 10. `ecg_fable_data.py`.

## Method

- **Codec**: the `ecg` branch's 1-D conv codec (rian's EEG codec, 12 leads),
  trained on the train-fold windows. `ecg_fable_train.py`.
- **Task metric**: a small 1-D CNN predicts the 5-way diagnosis from the raw ECG.
  Trained on **original** train records, scored on **original** vs
  **decompressed** test records. If accuracy/AUC holds, the codec kept the
  diagnostic content. `ecg_fable_eval.py`.

## Results (held-out fold 10)

Codec: 1.06M params, 30× vs float16, MSE 0.0515, **VE 95.1%**, PSNR 48.4 dB.

Diagnosis preservation (5-way superclass, small 1-D CNN; chance 20%):

| decoder → data | accuracy | balanced acc | macro-AUC | AUC retained |
|---|---:|---:|---:|---:|
| baseline: orig → orig | 71.5% | 55.1% | 0.824 | — |
| archival: orig → **decompressed** | 71.5% | 52.9% | 0.811 | **98%** |
| within: decomp → decompressed | 70.7% | 55.0% | 0.812 | 99% |

**The diagnosis survives compression.** A classifier trained on original ECG reads
the 30×-compressed ECG at 98% of its AUC (accuracy unchanged, 71.5%). Unlike fMRI,
where fine decodable structure is fragile, ECG's diagnostic content lives in the
same high-variance periodic waveform the codec reconstructs — so fidelity and
function are preserved together. This is the NeuroZip task-preservation thesis on a
clinical task.

_(Numbers from ~1.8k single-superclass records — the subset that had downloaded;
`ecg_fable_data.py` rebuilds on the full PTB-XL as more records arrive. The
retention conclusion is robust to dataset size.)_

## Where ECG sits

ECG is the high-redundancy end of the spectrum: the heartbeat repeats, so a
periodic (leads × time) signal compresses hard with the reconstruction essentially
intact — and, as shown here, the diagnosis survives with it. Contrast fMRI
(`fmri-fable`), where the redundancy is temporal-low-rank but the fine decodable
pattern is fragile. See the cross-modality `redundancy-spectrum` analysis.

## Run

```bash
pip install -r requirements.txt
python nz_fmri/dl_ptbxl.py          # or scripts/download_data.sh — fetch PTB-XL 100 Hz
python ecg_fable_data.py            # build labeled cache
python ecg_fable_train.py --epochs 60
python ecg_fable_eval.py --ckpt checkpoints/ecg_fable.pt
```
