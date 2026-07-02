# HAR-fable — the codec generalizes beyond biosignals

Branch `har-fable` (off `ecg`). To show the NeuroZip codec is a generic
`(channels × time)` compressor and not specific to neural data, we apply the exact
same 1-D conv codec to **wearable-sensor** data and ask the same
task-preservation question.

## Dataset

**UCI HAR** (smartphone inertial sensors): 9 channels (body/total accelerometer
x/y/z + gyroscope x/y/z) × 128 samples (2.56 s @ 50 Hz), **6 activities** (walking,
walking up/down, sitting, standing, laying), official train/test **subject** split.
`har_fable_data.py`. Per-channel z-score (train stats).

## Method

- **Codec**: `har_codec.py` — the same 1-D conv codec (rian's blocks), sized for a
  9-channel, 128-sample window (0.26M params).
- **Task metric**: a small 1-D CNN predicts the 6-way activity from the raw window,
  trained on **original** train, scored on **original** vs **decompressed** test.

## Results (held-out test subjects)

| metric | value |
|---|---|
| codec params | 0.26M |
| compression vs float16 | **35×** |
| reconstruction MSE / VE | 0.061 / **93.3%** |
| PSNR | 37.9 dB |

Activity recognition (6-way, chance 17%):

| decoder → data | accuracy | retained |
|---|---:|---:|
| baseline: orig → orig | 92.4% | — |
| archival: orig → **decompressed** | 90.8% | **98%** |
| within: decomp → decompressed | 92.0% | 100% |

**Activity recognition survives 35× compression** (98% of accuracy retained). The
same codec, unchanged, works on wearable sensors as well as on ECG — the method is
a general redundancy-matched time-series codec, and the task-preservation thesis
holds outside the brain. On the redundancy spectrum, periodic-motion sensor data
sits near ECG (high temporal + channel redundancy).

## Run

```bash
pip install -r requirements.txt
# download UCI HAR into data_har/ (see below), then:
python har_fable_data.py     # build cache
python har_fable.py          # train codec + classifier, report preservation
```

Download: `data_har/` — UCI HAR Dataset
(`https://archive.ics.uci.edu/static/public/240/`).
