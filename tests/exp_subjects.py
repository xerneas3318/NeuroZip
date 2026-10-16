"""Grant workstream 1: per-subject generalization + single-trial evaluation.

For each THINGS-EEG subject, independently:
  1. Train that subject's frozen judge (attention projector, InfoNCE to CLIP).
  2. Per rate tier: train a fidelity codec for E epochs, then branch it into
       fidelity-continued  (E more epochs, MSE + rate)            -- baseline
       NeuroZip            (E more epochs, MSE + rate + task)     -- ours
     Same architecture, same init, same step count; only the task term differs.
  3. Score every codec (and raw EEG = ceiling) on three test views:
       avg80  -- the 80 repetitions averaged (paper protocol, 200 queries)
       avg4   -- disjoint groups of 4 reps averaged (4,000 queries)
       single -- every single trial, no averaging (16,000 queries)
     against the 200-concept CLIP image and CLIP text galleries.

Results are dumped incrementally to tests/results/SUBJECTS.json, keyed by
subject, so a partial run (or a run split across invocations) still yields
stats. Subjects already present in the JSON are skipped unless --force.

Run:  .venv/bin/python tests/exp_subjects.py --subjects sub-01 sub-02 ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data as td
from clip_proj import EEGProjector, info_nce
from codec import EEGCodec

C_LAT, HIDDEN, N_CH, N_T = 32, 128, 63, 250
BPP_FACTOR = (C_LAT * 32) / (N_CH * N_T)           # bits/latent-symbol -> bits/sample
OUT = ROOT / "tests" / "results" / "SUBJECTS.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# (tag, lambda_rate). Spans the shipped v4 low..high range.
TIERS = [("low", 0.05), ("med", 0.01), ("high", 0.002)]
LAMBDA_TASK = 3.0


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def batches(n, batch, shuffle, gen=None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, batch):
        yield idx[i:i + batch]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_subject(subject):
    tr = td.ThingsEEG(split="train", subject=subject)
    te = td.ThingsEEG(split="test", subject=subject)
    _, te_text, _ = te.trial_averaged()
    text_gallery, concepts = te.concept_clip_text()
    cidx = {c: i for i, c in enumerate(concepts)}
    # trial_averaged row i -> text-gallery index (image gallery is row-aligned)
    row2txt = torch.tensor([cidx[str(t)] for t in te_text]).long()

    views = {}
    avg80, _, _ = te.trial_averaged()
    views["avg80"] = (avg80, torch.arange(te.n_concepts))
    views["avg4"] = te.k_averaged(4, seed=0)
    views["single"] = te.single_trial()
    d = {
        # GPU-resident fp16 copy of train: 66k x 63 x 250 = 2.1 GB
        "x_train": torch.from_numpy(tr.eeg).half().to(DEV),
        "t_train": torch.from_numpy(tr.clip_img).float().to(DEV),
        "views": {k: (x.float(), row) for k, (x, row) in views.items()},
        "img_gallery": te.concept_clip_img().float(),
        "txt_gallery": text_gallery.float(),
        "row2txt": row2txt,
        "meta": {"train_epochs": len(tr.eeg), "test_concepts": te.n_concepts,
                 "test_reps": te.n_reps,
                 "n_queries": {k: int(v[0].shape[0]) for k, v in views.items()}},
    }
    del tr, te
    return d


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def topk(emb, gallery, gold, ks=(1, 5)):
    sims = F.normalize(emb, dim=-1) @ F.normalize(gallery, dim=-1).t()
    ranks = sims.topk(max(ks), dim=-1).indices
    return {f"top{k}": round((ranks[:, :k] == gold[:, None]).any(1).float().mean().item(), 4)
            for k in ks}


def retrieval(emb, row, d):
    return {"image": topk(emb, d["img_gallery"], row),
            "text": topk(emb, d["txt_gallery"], d["row2txt"][row])}


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #

def train_judge(d, epochs, batch=256, log=print):
    proj = EEGProjector(hidden=HIDDEN, n_attn=2).to(DEV).train()
    opt = torch.optim.AdamW(proj.parameters(), lr=5e-4, weight_decay=1e-4)
    n = d["x_train"].shape[0]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * math.ceil(n / batch))
    for ep in range(epochs):
        for b in batches(n, batch, shuffle=True):
            loss = info_nce(proj(d["x_train"][b].float()), d["t_train"][b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)   # as train.py
            opt.step(); sched.step()
        r = retrieval(embed(proj, d["views"]["avg80"][0]), d["views"]["avg80"][1], d)
        log(f"  [judge] ep {ep+1}/{epochs} avg80 img top1={r['image']['top1']:.3f} "
            f"top5={r['image']['top5']:.3f}")
    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


@torch.no_grad()
def embed(proj, x, codec=None, batch=512):
    """Project raw EEG (codec=None) or its codec reconstruction. Returns
    (embeddings, mse_per_sample, bits_per_latent_symbol)."""
    was = proj.training
    proj.eval()
    out, se, bits, n = [], 0.0, 0.0, 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].to(DEV)
        if codec is not None:
            yq = torch.round(codec.encoder(xb))
            xh = codec.decoder(yq)
            se += F.mse_loss(xh, xb, reduction="sum").item()
            bits += codec.prior.bits(yq).item() * xb.shape[0]
            xb = xh
        out.append(proj(xb).cpu()); n += xb.shape[0]
    proj.train(was)
    emb = torch.cat(out)
    if codec is None:
        return emb
    return emb, se / (n * N_CH * N_T), bits / n


# --------------------------------------------------------------------------- #
# Held-out classifier (non-circular control, --holdout)
# --------------------------------------------------------------------------- #

def holdout_split(d, R=80):
    """Same split as train.py `classifier`: train on 60 test reps, score on 20."""
    x1, _ = d["views"]["single"]
    perm = np.random.default_rng(0).permutation(R)
    x = x1.reshape(-1, R, N_CH, N_T)
    tr, va = x[:, perm[:60]], x[:, perm[60:]]
    n = x.shape[0]
    return tr, {"single_heldout_reps": (va.reshape(-1, N_CH, N_T), torch.arange(n).repeat_interleave(20)),
                "avg20_heldout_reps": (va.mean(1), torch.arange(n))}


def train_holdout(tr, steps=4000, batch=256, K=5, log=print):
    """Concept classifier on raw EEG only (never sees a codec). Each sample is
    the average of K random training reps of a random concept (train.py recipe)."""
    from train import HoldoutClassifier
    n, r = tr.shape[:2]
    trg = tr.half().to(DEV)
    clf = HoldoutClassifier(n_classes=n, hidden=HIDDEN, n_attn=2).to(DEV).train()
    opt = torch.optim.AdamW(clf.parameters(), lr=3e-4, weight_decay=1e-4)
    for _ in range(steps):
        c = torch.randint(0, n, (batch,), device=DEV)
        reps = torch.rand(batch, r, device=DEV).argsort(1)[:, :K]
        xb = trg[c[:, None], reps].float().mean(1)
        loss = F.cross_entropy(clf(xb), c)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
        opt.step()
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    del trg
    return clf


@torch.no_grad()
def holdout_acc(clf, x, row, codec=None, batch=512):
    hit = 0
    for i in range(0, x.shape[0], batch):
        xb = x[i:i + batch].to(DEV)
        if codec is not None:
            xb = codec.decoder(torch.round(codec.encoder(xb)))
        hit += (clf(xb).argmax(-1).cpu() == row[i:i + batch]).sum().item()
    return round(hit / x.shape[0], 4)


# --------------------------------------------------------------------------- #
# Codecs
# --------------------------------------------------------------------------- #

def train_codec(codec, d, proj, lam_rate, lam_task, epochs, idx, batch=256, lr=2e-3):
    codec.train()
    opt = torch.optim.Adam(codec.parameters(), lr=lr)
    n = idx.numel()
    for _ in range(epochs):
        for b in batches(n, batch, shuffle=True):
            sel = idx[b]
            x = d["x_train"][sel].float()
            xh, bps, _ = codec(x)
            loss = F.mse_loss(xh, x) + lam_rate * bps * BPP_FACTOR
            if lam_task > 0:
                tgt = F.normalize(d["t_train"][sel], dim=-1)
                loss = loss + lam_task * (1 - F.cosine_similarity(proj(xh), tgt, dim=-1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    codec.eval()
    return codec


def score_codec(codec, proj, d, clf=None):
    out = {}
    if clf is not None:
        for v, (x, row) in d["holdout_views"].items():
            out[v] = {"holdout_top1": holdout_acc(clf, x, row, codec)}
    for view, (x, row) in d["views"].items():
        emb, mse, bps = embed(proj, x, codec)
        out[view] = {"mse": round(mse, 5), "retrieval": retrieval(emb, row, d)}
        if view == "avg80":
            out["bpp"] = round(bps * BPP_FACTOR, 4)
            out["ratio_vs_fp16"] = round(16 / (bps * BPP_FACTOR), 1)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run_subject(subject, args, log):
    t0 = time.time()
    seed_all(args.seed)
    d = load_subject(subject)
    log(f"[{subject}] loaded {d['meta']} in {time.time()-t0:.0f}s")
    res = {"meta": d["meta"], "seed": args.seed}

    proj = train_judge(d, args.judge_epochs, log=log)
    res["ceiling_raw"] = {v: retrieval(embed(proj, x), row, d) for v, (x, row) in d["views"].items()}
    clf = None
    if args.holdout:
        tr, d["holdout_views"] = holdout_split(d)
        clf = train_holdout(tr, log=log)
        del tr
        for v, (x, row) in d["holdout_views"].items():
            res["ceiling_raw"][v] = {"holdout_top1": holdout_acc(clf, x, row)}
    log(f"[{subject}] ceiling {json.dumps(res['ceiling_raw'])}")

    n = d["x_train"].shape[0]
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(n, generator=g)[: min(args.train_cap, n)].to(DEV)
    res["codec_train_epochs"] = int(idx.numel())

    res["tiers"] = {}
    for tag, lam_rate in TIERS:
        seed_all(args.seed)
        base = train_codec(EEGCodec(c_lat=C_LAT, hidden=HIDDEN, n_attn=0).to(DEV),
                           d, proj, lam_rate, 0.0, args.codec_epochs, idx)
        state = {k: v.clone() for k, v in base.state_dict().items()}
        fid = train_codec(base, d, proj, lam_rate, 0.0, args.codec_epochs, idx)
        nz = EEGCodec(c_lat=C_LAT, hidden=HIDDEN, n_attn=0).to(DEV)
        nz.load_state_dict(state)
        nz = train_codec(nz, d, proj, lam_rate, LAMBDA_TASK, args.codec_epochs, idx)
        res["tiers"][tag] = {"lambda_rate": lam_rate,
                             "fidelity": score_codec(fid, proj, d, clf),
                             "neurozip": score_codec(nz, proj, d, clf)}
        f, z = res["tiers"][tag]["fidelity"], res["tiers"][tag]["neurozip"]
        hl = (f" | holdout single fid={f['single_heldout_reps']['holdout_top1']:.3f} "
              f"nz={z['single_heldout_reps']['holdout_top1']:.3f}") if clf is not None else ""
        log(f"[{subject}] {tag}: bpp fid={f['bpp']} nz={z['bpp']}{hl} | "
            + " | ".join(f"{v} img top5 fid={f[v]['retrieval']['image']['top5']:.3f} "
                         f"nz={z[v]['retrieval']['image']['top5']:.3f}"
                         for v in d["views"]))
    res["secs"] = round(time.time() - t0, 1)
    del d
    torch.cuda.empty_cache()
    return res


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    a.add_argument("--judge_epochs", type=int, default=12)
    a.add_argument("--codec_epochs", type=int, default=6)
    a.add_argument("--train_cap", type=int, default=32000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", type=str, default=str(OUT))
    a.add_argument("--force", action="store_true")
    a.add_argument("--holdout", action="store_true",
                   help="also train a per-subject concept classifier on 60 test reps "
                        "and score codecs on the 20 unseen reps (non-circular control)")
    args = a.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    allres = json.loads(out.read_text()) if out.exists() else {}
    allres["config"] = {"judge_epochs": args.judge_epochs, "codec_epochs": args.codec_epochs,
                        "train_cap": args.train_cap, "tiers": TIERS, "lambda_task": LAMBDA_TASK,
                        "holdout": args.holdout,
                        "codec": f"conv-only c_lat={C_LAT} hidden={HIDDEN}",
                        "judge": f"attn n_attn=2 hidden={HIDDEN}",
                        "protocol": "fidelity trained E epochs, then branched: fidelity +E "
                                    "(no task) vs NeuroZip +E (task); same init and steps"}
    log = lambda s: print(s, flush=True)
    for s in args.subjects:
        if s in allres.get("subjects", {}) and not args.force:
            log(f"[{s}] already in {out}; skipping"); continue
        if not (td.EEG_DIR / s / "test.pt").exists():
            log(f"[{s}] data missing; skipping"); continue
        allres.setdefault("subjects", {})[s] = run_subject(s, args, log)
        out.write_text(json.dumps(allres, indent=1))
        log(f"[{s}] done in {allres['subjects'][s]['secs']}s -> {out}")


if __name__ == "__main__":
    main()
