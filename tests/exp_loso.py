"""Grant workstream 1: leave-one-subject-out (LOSO) cross-subject evaluation.

For each held-out subject S:
  * Pool the train split of the other 9 subjects (a random --per_subject
    epochs from each, so the pool fits on one GPU) and train the frozen judge
    and the fidelity / NeuroZip codec pair on it, exactly as exp_subjects.py.
  * Evaluate on S's test split only (avg80 / avg4 / single views). Neither the
    judge nor the codecs ever see S's EEG.

Normalization: every subject (including S) is z-scored with per-channel stats
from its own *train* split. For S that is unlabeled calibration data, the same
thing a deployed codec would compute from a new wearer's first recordings; no
labels or test data are used.

Results -> tests/results/LOSO.json, keyed by held-out subject.

Run:  .venv/bin/python tests/exp_loso.py --heldout sub-01 sub-02 ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_subjects as E
import data as td

ALL = [f"sub-{i:02d}" for i in range(1, 11)]
OUT = E.ROOT / "tests" / "results" / "LOSO.json"


def load_fold(heldout, per_subject, log):
    xs, ts = [], []
    for s in ALL:
        if s == heldout:
            continue
        tr = td.ThingsEEG(split="train", subject=s, max_epochs=per_subject)
        xs.append(torch.from_numpy(tr.eeg).half()); ts.append(torch.from_numpy(tr.clip_img))
        del tr
    # Views + galleries for the held-out subject come from the per-subject loader,
    # but its train split is discarded (only its norm stats are used).
    d = E.load_subject(heldout)
    d["x_train"] = torch.cat(xs).to(E.DEV)
    d["t_train"] = torch.cat(ts).float().to(E.DEV)
    d["meta"]["train_epochs"] = int(d["x_train"].shape[0])
    d["meta"]["train_subjects"] = [s for s in ALL if s != heldout]
    log(f"[loso {heldout}] pooled {d['meta']['train_epochs']} epochs from 9 subjects")
    return d


def run_fold(heldout, args, log):
    t0 = time.time()
    E.seed_all(args.seed)
    d = load_fold(heldout, args.per_subject, log)
    res = {"meta": d["meta"], "seed": args.seed}
    proj = E.train_judge(d, args.judge_epochs, log=log)
    res["ceiling_raw"] = {v: E.retrieval(E.embed(proj, x), row, d)
                          for v, (x, row) in d["views"].items()}
    log(f"[loso {heldout}] ceiling {json.dumps(res['ceiling_raw'])}")

    n = d["x_train"].shape[0]
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(n, generator=g)[: min(args.train_cap, n)].to(E.DEV)
    res["codec_train_epochs"] = int(idx.numel())
    res["tiers"] = {}
    for tag, lam_rate in E.TIERS:
        E.seed_all(args.seed)
        base = E.train_codec(E.EEGCodec(c_lat=E.C_LAT, hidden=E.HIDDEN, n_attn=0).to(E.DEV),
                             d, proj, lam_rate, 0.0, args.codec_epochs, idx)
        state = {k: v.clone() for k, v in base.state_dict().items()}
        fid = E.train_codec(base, d, proj, lam_rate, 0.0, args.codec_epochs, idx)
        nz = E.EEGCodec(c_lat=E.C_LAT, hidden=E.HIDDEN, n_attn=0).to(E.DEV)
        nz.load_state_dict(state)
        nz = E.train_codec(nz, d, proj, lam_rate, E.LAMBDA_TASK, args.codec_epochs, idx)
        res["tiers"][tag] = {"lambda_rate": lam_rate,
                             "fidelity": E.score_codec(fid, proj, d),
                             "neurozip": E.score_codec(nz, proj, d)}
        f, z = res["tiers"][tag]["fidelity"], res["tiers"][tag]["neurozip"]
        log(f"[loso {heldout}] {tag}: bpp fid={f['bpp']} nz={z['bpp']} | "
            + " | ".join(f"{v} img top5 fid={f[v]['retrieval']['image']['top5']:.3f} "
                         f"nz={z[v]['retrieval']['image']['top5']:.3f}" for v in d["views"]))
    res["secs"] = round(time.time() - t0, 1)
    del d
    torch.cuda.empty_cache()
    return res


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--heldout", nargs="+", default=ALL)
    a.add_argument("--per_subject", type=int, default=8000)
    a.add_argument("--judge_epochs", type=int, default=12)
    a.add_argument("--codec_epochs", type=int, default=6)
    a.add_argument("--train_cap", type=int, default=32000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", type=str, default=str(OUT))
    a.add_argument("--force", action="store_true")
    args = a.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres["config"] = {"per_subject": args.per_subject, "judge_epochs": args.judge_epochs,
                        "codec_epochs": args.codec_epochs, "train_cap": args.train_cap,
                        "tiers": E.TIERS, "lambda_task": E.LAMBDA_TASK,
                        "normalization": "per-subject train-split channel stats (incl. held-out)"}
    log = lambda s: print(s, flush=True)
    missing = [s for s in ALL if not (td.EEG_DIR / s / "train.pt").exists()]
    if missing:
        raise SystemExit(f"LOSO needs all 10 subjects; missing {missing}")
    for s in args.heldout:
        if s in allres.get("folds", {}) and not args.force:
            log(f"[loso {s}] already in {out}; skipping"); continue
        allres.setdefault("folds", {})[s] = run_fold(s, args, log)
        out.write_text(json.dumps(allres, indent=1))
        log(f"[loso {s}] done in {allres['folds'][s]['secs']}s -> {out}")


if __name__ == "__main__":
    main()
