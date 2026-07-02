"""Group A / objective: A6 contrastive task loss vs cosine-to-target.

NeuroZip's task term is 1 - cos(P(EEG_hat), CLIP_target). That only pulls the
reconstruction toward its own target; it never explicitly pushes it AWAY from
other concepts. The retrieval metric it's evaluated on is exactly a
discrimination task. So we test an InfoNCE contrastive task loss: for each
reconstruction, cross-entropy over similarity to the full concept-prototype
gallery (pull to own concept, push off all others).

Protocol (matched rate, same frozen judge):
  1. Train projector P on EEG->prototype (frozen afterward).
  2. Train two task-aware codecs from the same init, same lambda_rate:
       cosine     : L = mse + lam_rate*bps + lam_task*(1 - cos)
       contrastive: L = mse + lam_rate*bps + lam_task*CE(P(hat)@proto.T/tau, y)
  3. Compare top-1/5/10 retrieval of reconstructed test EEG against the
     prototype gallery, plus bpp and MSE.

Run:  .venv/bin/python tests/exp_contrastive.py
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import exp_common as C
from codec import quantize

TAU = 0.07
LAM_RATE = 0.06        # aggressive rate so fidelity retrieval sits below ceiling
LAM_TASK = 2.0
LAM_RECON = 1.0


def train_projector(data, dev, epochs=25, batch=64):
    """Frozen judge: EEG -> prototype space, trained CLIP-style over the gallery."""
    proj = C.make_projector(hidden=32).to(dev).train()
    proto = data["prototypes"].to(dev)
    opt = torch.optim.Adam(proj.parameters(), lr=2e-3)
    for _ in range(epochs):
        for x, y in C.iterate(data["eeg_train"], data["lab_train"], batch=batch,
                              shuffle=True, dev=dev):
            emb = proj(x)
            logits = emb @ F.normalize(proto, dim=-1).t() / TAU
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    proj.eval()
    for p in proj.parameters():
        p.requires_grad_(False)
    return proj


class TaskCodec(nn.Module):
    def __init__(self):
        super().__init__()
        c = C.make_codec(**C.TINY)
        self.enc, self.dec, self.prior = c.encoder, c.decoder, c.prior

    def forward(self, x):
        y = self.enc(x); yq = quantize(y, self.training)
        return self.dec(yq), self.prior.bits(yq)


def train_task_codec(mode, data, dev, proj, init_state, lam_rate, lam_task,
                     epochs=18, batch=64):
    m = TaskCodec().to(dev)
    m.load_state_dict(init_state)
    m.train()
    proto = F.normalize(data["prototypes"].to(dev), dim=-1)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(epochs):
        for x, y, tgt in C.iterate(data["eeg_train"], data["lab_train"],
                                   data["tgt_train"], batch=batch, shuffle=True, dev=dev):
            xh, bps = m(x)
            loss = LAM_RECON * F.mse_loss(xh, x) + lam_rate * bps
            if mode != "fidelity":
                emb = proj(xh)                      # grad flows through frozen P
                if mode == "cosine":
                    task = (1 - F.cosine_similarity(emb, F.normalize(tgt, dim=-1), dim=-1)).mean()
                else:  # contrastive
                    task = F.cross_entropy(emb @ proto.t() / TAU, y)
                loss = loss + lam_task * task
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def _interp(xs, ys, x):
    pts = sorted(zip(xs, ys))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            f = (x - xs[i]) / (xs[i + 1] - xs[i] + 1e-9)
            return ys[i] + f * (ys[i + 1] - ys[i])
    return ys[-1]


@torch.no_grad()
def eval_codec(m, data, dev, proj, batch=128):
    m.eval()
    proto = data["prototypes"]
    embs, mse, bps, n = [], 0.0, 0.0, 0
    for (x,) in C.iterate(data["eeg_test"], batch=batch, shuffle=False, dev=dev):
        y = m.enc(x); yq = torch.round(y)
        xh = m.dec(yq)
        embs.append(proj(xh).cpu())
        mse += F.mse_loss(xh, x, reduction="sum").item()
        bps += m.prior.bits(yq).item() * x.shape[0]
        n += x.shape[0]
    emb = torch.cat(embs, 0)
    ret = C.retrieval_topk(emb, proto, data["lab_test"])
    return {"bpp": round(C.latent_bpp(bps / n, C.TINY["c_lat"]), 4),
            "ratio": round(C.ratio_vs_fp16(C.latent_bpp(bps / n, C.TINY["c_lat"])), 1),
            "mse": round(mse / (n * C.N_CH * C.N_T), 4), **ret}


def main():
    C.seed_all(0)
    dev = C.device()
    C.banner(f"A6  contrastive vs cosine task loss  (device={dev})")
    # task-fragile regime: weak concept signal buried under a high-variance,
    # concept-independent distractor, so MSE-only compression sacrifices the
    # discriminative signal (the regime where task-aware training matters). Mild
    # enough that fidelity retains PARTIAL retrieval, leaving room to beat it.
    data = C.make_dataset(n_concepts=80, trials_per_concept=30, noise=0.3,
                          task_scale=0.7, distractor_scale=1.2, seed=0)

    proj = train_projector(data, dev)

    # no-compression ceiling: judge on raw (uncompressed) test EEG
    with torch.no_grad():
        raw_emb = []
        for (x,) in C.iterate(data["eeg_test"], batch=128, shuffle=False, dev=dev):
            raw_emb.append(proj(x).cpu())
        ceiling = C.retrieval_topk(torch.cat(raw_emb, 0), data["prototypes"], data["lab_test"])
    init_state = {k: v.clone() for k, v in TaskCodec().state_dict().items()}

    # Sweep lambda_rate per mode -> (bpp, top5) curves; compare task modes to
    # fidelity at MATCHED bpp via interpolation (removes the rate confound).
    lams = (0.004, 0.012, 0.035, 0.08)
    curves = {}
    for mode in ("fidelity", "cosine", "contrastive"):
        pts = []
        for lam in lams:
            m = train_task_codec(mode, data, dev, proj, init_state,
                                 lam_rate=lam, lam_task=LAM_TASK)
            pts.append(eval_codec(m, data, dev, proj))
        curves[mode] = pts

    def matched(mode):
        tb = [p["bpp"] for p in curves[mode]]
        t5 = [p["top5"] for p in curves[mode]]
        t1 = [p["top1"] for p in curves[mode]]
        out = []
        for p in curves["fidelity"]:
            if min(tb) <= p["bpp"] <= max(tb):
                out.append({"bpp": p["bpp"], "ratio": p["ratio"],
                            "fidelity_top5": p["top5"],
                            f"{mode}_top5": round(_interp(tb, t5, p["bpp"]), 4),
                            "top5_delta_pp": round(100 * (_interp(tb, t5, p["bpp"]) - p["top5"]), 2),
                            "fidelity_top1": p["top1"],
                            f"{mode}_top1": round(_interp(tb, t1, p["bpp"]), 4)})
        return out

    results = {
        "no_compression_ceiling": ceiling,
        "chance_top1": round(1 / data["n_concepts"], 4),
        "fidelity_curve": curves["fidelity"],
        "cosine_curve": curves["cosine"],
        "contrastive_curve": curves["contrastive"],
        "cosine_vs_fidelity_matched_bpp": matched("cosine"),
        "contrastive_vs_fidelity_matched_bpp": matched("contrastive"),
    }
    print("RESULTS_JSON:" + json.dumps(results))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
