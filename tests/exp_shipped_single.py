"""Does the shipped v4 result (hackathon headline, sub-01) survive without
trial averaging?

Scores the shipped fidelity_v4_* / neurozip_v4_* checkpoints with the shipped
frozen judge on sub-01's avg80 / avg4 / single-trial test views, plus the
held-out concept classifier (independent of the judge).

The shipped classifier (train.py `classifier`) is trained on 60 of the 80 test
repetitions per concept (np.random.default_rng(0).permutation(80)[:60]), so it
is scored ONLY on the other 20 repetitions: single trials (200 x 20 = 4,000)
and their 20-rep average (200). Scoring it on all 80 reps would leak.

Run:  .venv/bin/python tests/exp_shipped_single.py [--ckpt ~/.neurozip/checkpoints]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_subjects as E
from clip_proj import load_frozen_projector
from codec import EEGCodec
from train import HoldoutClassifier

TIERS = ("low", "med", "high", "xhigh")


def load_codec(path):
    st = torch.load(path, weights_only=False, map_location=E.DEV)
    cfg = st.get("config", {})
    m = EEGCodec(c_lat=cfg.get("c_lat", 32), hidden=cfg.get("hidden", 128),
                 n_attn=cfg.get("n_attn", 0)).to(E.DEV)
    m.load_state_dict(st["model"]); m.eval()
    return m


def load_holdout(path):
    st = torch.load(path, weights_only=False, map_location=E.DEV)
    m = HoldoutClassifier(n_classes=st["n_classes"], hidden=st["hidden"],
                          n_attn=st.get("n_attn", 0), attn_heads=st.get("attn_heads", 4)).to(E.DEV)
    m.load_state_dict(st["model"]); m.eval()
    return m


@torch.no_grad()
def holdout_acc(clf, x, row, codec=None, batch=512):
    hit = 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].to(E.DEV)
        if codec is not None:
            xb = codec.decoder(torch.round(codec.encoder(xb)))
        hit += (clf(xb).argmax(-1).cpu() == row[i:i + batch]).sum().item()
    return round(hit / x.shape[0], 4)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", default=str(Path.home() / ".neurozip" / "checkpoints"))
    args = a.parse_args()
    ck = Path(args.ckpt)
    d = E.load_subject("sub-01")
    del d["x_train"], d["t_train"]
    torch.cuda.empty_cache()
    judge = load_frozen_projector(ck / "clip_proj.pt", E.DEV)
    clf = load_holdout(ck / "holdout_classifier.pt") if (ck / "holdout_classifier.pt").exists() else None

    # classifier views: only the 20 reps it never trained on
    x1, row1 = d["views"]["single"]
    R = 80
    va = np.random.default_rng(0).permutation(R)[60:]
    mask = torch.from_numpy(np.isin(np.arange(x1.shape[0]) % R, va))
    cv = {"single_heldout_reps": (x1[mask], row1[mask]),
          "avg20_heldout_reps": (x1.reshape(-1, R, *x1.shape[1:])[:, va].mean(1),
                                 torch.arange(x1.shape[0] // R))}

    res = {"ceiling_raw": {}, "holdout_note": "classifier scored on the 20 reps it never trained on"}
    for v, (x, row) in d["views"].items():
        res["ceiling_raw"][v] = E.retrieval(E.embed(judge, x), row, d)
    if clf is not None:
        for v, (x, row) in cv.items():
            res["ceiling_raw"][v] = {"holdout_top1": holdout_acc(clf, x, row)}
    print("raw", json.dumps(res["ceiling_raw"]), flush=True)
    for t in TIERS:
        for kind in ("fidelity", "neurozip"):
            p = ck / f"{kind}_v4_{t}.pt"
            if not p.exists():
                continue
            m = load_codec(p)
            r = E.score_codec(m, judge, d)
            if clf is not None:
                for v, (x, row) in cv.items():
                    r[v] = {"holdout_top1": holdout_acc(clf, x, row, m)}
            res.setdefault(t, {})[kind] = r
            print(t, kind, r["bpp"], {v: r[v]["retrieval"]["image"]["top5"] for v in d["views"]},
                  {v: r[v]["holdout_top1"] for v in cv if v in r}, flush=True)
    out = E.ROOT / "tests" / "results" / "SHIPPED_single_trial.json"
    out.write_text(json.dumps(res, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
