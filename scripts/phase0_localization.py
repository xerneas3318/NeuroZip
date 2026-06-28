"""Phase 0 — does object identity in EEG actually localize in space and time?

Three checks on already-trained v4 checkpoints. If any of them show
strong concentration, we have a real biological story to tell (Phase 1).
If all three look flat, the biology reframe is rhetoric.

Outputs:
  - prints a verdict table to stdout
  - writes plots/phase0_*.png so we can eyeball
  - writes plots/phase0_summary.json with the numerical evidence

Checks
------
1. per-channel held-out ablation
     Zero each channel one-at-a-time on the decompressed test EEG.
     Measure top-1 drop in the held-out classifier. Top channels =
     where object identity is concentrated. Compare to known visual-
     cortex sites (O1 O2 PO7 PO8) for biological validation.

2. per-timepoint reconstruction MSE
     Plot MSE(raw, recon) along the 250-timestep axis for fidelity vs
     NeuroZip. Annotate ERP windows: P100 (~90-130 ms), N170 (~150-200 ms),
     P200 (~180-250 ms), P300 (~250-400 ms). If NeuroZip preserves the
     N170/P300 windows tighter than fidelity does, that's the "task loss
     keeps the discriminative slice" claim with evidence.

3. per-channel reconstruction MSE
     Same idea on the channel axis. If NeuroZip concentrates preserved
     fidelity in occipitotemporal channels while fidelity scatters error
     uniformly across the scalp, that's the strongest possible spatial
     story.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import ThingsEEG, N_CHANNELS, N_TIMES
from clip_proj import load_frozen_projector
from codec import EEGCodec, CHECKPOINTS
from train import HoldoutClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# THINGS-EEG channel names (63 ch, 10-10 layout, ch_names from sub-01/test.pt).
# Pulled from the dataset's ch_names array.
CH_NAMES = ['Fp1','Fp2','AF7','AF3','AFz','AF4','AF8','F7','F5','F3','F1','Fz',
            'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6',
            'FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3',
            'CP1','CPz','CP2','CP4','CP6','TP8','P9','P7','P5','P3','P1','Pz','P2',
            'P4','P6','P8','P10','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz']

# Visual cortex / object-recognition channels we EXPECT to dominate.
EXPECTED_VISUAL = {'O1','O2','Oz','Iz','PO7','PO8','PO3','PO4','POz','P7','P8','P9','P10'}

# Standard visual-evoked ERP components (ms post-stimulus onset).
ERP_WINDOWS = {
    "P100": (90, 130),
    "N170": (150, 200),
    "P200": (200, 260),
    "P300": (280, 400),
}


def load_codec(name: str) -> EEGCodec:
    path = CHECKPOINTS / f"{name}.pt"
    state = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = state.get("config", {})
    m = EEGCodec(c_lat=cfg.get("c_lat", 32), hidden=cfg.get("hidden", 128),
                  n_attn=cfg.get("n_attn", 0)).to(DEVICE)
    m.load_state_dict(state["model"]); m.eval()
    return m


def load_holdout():
    p = CHECKPOINTS / "holdout_classifier.pt"
    st = torch.load(p, weights_only=False, map_location=DEVICE)
    clf = HoldoutClassifier(n_classes=st["n_classes"], hidden=st["hidden"],
                              n_attn=st.get("n_attn", 0),
                              attn_heads=st.get("attn_heads", 4)).to(DEVICE)
    clf.load_state_dict(st["model"]); clf.eval()
    return clf


@torch.no_grad()
def decompress(codec: EEGCodec, eeg: torch.Tensor) -> torch.Tensor:
    """eeg: (N, 63, 250) normalized. Returns reconstructed (N, 63, 250) normalized."""
    xs = []
    for i in range(0, eeg.size(0), 200):
        xh, _ = codec.compress_then_reconstruct(eeg[i:i+200].to(DEVICE))
        xs.append(xh.cpu())
    return torch.cat(xs, dim=0)


# =============================================================================
# Check 1 — per-channel ablation of the held-out classifier
# =============================================================================

@torch.no_grad()
def check1_channel_ablation(codec_name: str, eeg_avg: torch.Tensor,
                             clf, base_label: str) -> dict:
    """Zero each channel; measure top-1 drop on the held-out classifier."""
    codec = load_codec(codec_name) if codec_name != "raw" else None
    if codec is None:
        recon = eeg_avg.clone().to(DEVICE)
    else:
        recon = decompress(codec, eeg_avg).to(DEVICE)

    gold = torch.arange(recon.size(0), device=DEVICE)
    base_acc = (clf(recon).argmax(-1) == gold).float().mean().item()

    drops = np.zeros(N_CHANNELS, dtype=np.float32)
    for c in range(N_CHANNELS):
        x = recon.clone()
        x[:, c, :] = 0.0
        acc = (clf(x).argmax(-1) == gold).float().mean().item()
        drops[c] = base_acc - acc          # bigger = more important

    # Concentration metrics
    sorted_drops = np.sort(drops)[::-1]
    cum = np.cumsum(sorted_drops) / max(sorted_drops.sum(), 1e-9)
    n_for_50 = int((cum >= 0.5).argmax() + 1) if cum.max() >= 0.5 else N_CHANNELS
    n_for_80 = int((cum >= 0.8).argmax() + 1) if cum.max() >= 0.8 else N_CHANNELS

    # How many of the top-15 channels are known visual-cortex sites?
    top_idx = np.argsort(-drops)[:15]
    top_names = [CH_NAMES[i] for i in top_idx]
    n_visual_in_top15 = sum(1 for n in top_names if n in EXPECTED_VISUAL)

    print(f"\n[{base_label}]")
    print(f"  base top-1 = {base_acc*100:.1f}%   ablation budget = {drops.sum()*100:.1f} pp")
    print(f"  channels needed to explain 50% of drop: {n_for_50}/63")
    print(f"  channels needed to explain 80% of drop: {n_for_80}/63")
    print(f"  top-15 ablation-sensitive channels: {top_names}")
    print(f"  of which known visual-cortex: {n_visual_in_top15}/15  (expected ≥6 for a spatial story)")
    return {"base_acc": base_acc,
            "drops_per_channel": drops.tolist(),
            "ch_names": CH_NAMES,
            "channels_for_50pct_of_drop": n_for_50,
            "channels_for_80pct_of_drop": n_for_80,
            "top15": top_names,
            "n_visual_in_top15": n_visual_in_top15}


# =============================================================================
# Check 2 — per-timepoint reconstruction MSE
# =============================================================================

@torch.no_grad()
def check2_per_timepoint_mse(codec_names: list[str], eeg_avg: torch.Tensor) -> dict:
    """MSE per timepoint, averaged across channels + epochs."""
    out = {}
    for name in codec_names:
        codec = load_codec(name)
        recon = decompress(codec, eeg_avg)
        err = (eeg_avg - recon).pow(2).mean(dim=(0, 1)).numpy()      # (250,)
        out[name] = err.tolist()
    return out


# =============================================================================
# Check 3 — per-channel reconstruction MSE
# =============================================================================

@torch.no_grad()
def check3_per_channel_mse(codec_names: list[str], eeg_avg: torch.Tensor) -> dict:
    out = {}
    for name in codec_names:
        codec = load_codec(name)
        recon = decompress(codec, eeg_avg)
        err = (eeg_avg - recon).pow(2).mean(dim=(0, 2)).numpy()      # (63,)
        out[name] = err.tolist()
    return out


# =============================================================================
# Plot helpers
# =============================================================================

def plot_channel_ablation(check1: dict, label: str, path: Path):
    drops = np.array(check1["drops_per_channel"])
    fig, ax = plt.subplots(figsize=(11, 3.2))
    colors = ['#d62728' if CH_NAMES[i] in EXPECTED_VISUAL else '#7c8497' for i in range(N_CHANNELS)]
    ax.bar(range(N_CHANNELS), drops * 100, color=colors)
    ax.set_xticks(range(N_CHANNELS)); ax.set_xticklabels(CH_NAMES, rotation=90, fontsize=7)
    ax.set_ylabel("top-1 drop when channel zeroed (pp)")
    ax.set_title(f"per-channel ablation — {label}   (red = expected visual cortex)")
    ax.axhline(0, color='#888', lw=0.6)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_per_timepoint(check2: dict, path: Path):
    t_ms = np.linspace(-200, 996, N_TIMES)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    style = {"fidelity": ("#7c8497", "-"), "neurozip": ("#d62728", "-")}
    for name, err in check2.items():
        kind = "neurozip" if "neurozip" in name else "fidelity"
        col, ls = style[kind]
        ax.plot(t_ms, err, color=col, ls=ls, lw=1.4, label=name, alpha=0.85)
    ymax = ax.get_ylim()[1]
    for label_name, (a, b) in ERP_WINDOWS.items():
        ax.axvspan(a, b, alpha=0.10, color="#1f9d55")
        ax.text((a+b)/2, ymax*0.97, label_name, ha="center", va="top",
                fontsize=9, color="#1f9d55", fontweight="bold")
    ax.axvline(0, color="#444", lw=0.7)
    ax.set_xlabel("time after stimulus (ms)")
    ax.set_ylabel("MSE  (channel- and epoch-averaged)")
    ax.set_title("per-timepoint reconstruction MSE  ·  green = visual ERP windows")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_per_channel_mse(check3: dict, path: Path):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    width = 0.4
    xs = np.arange(N_CHANNELS)
    items = list(check3.items())
    for i, (name, err) in enumerate(items):
        col = "#d62728" if "neurozip" in name else "#7c8497"
        ax.bar(xs + (i - 0.5) * width, err, width, color=col, alpha=0.85, label=name)
    # Highlight expected visual cortex tick labels
    for i, c in enumerate(CH_NAMES):
        if c in EXPECTED_VISUAL:
            ax.axvline(i, color="#1f9d55", alpha=0.15, lw=8, zorder=-1)
    ax.set_xticks(xs); ax.set_xticklabels(CH_NAMES, rotation=90, fontsize=7)
    ax.set_ylabel("MSE  (epoch- and time-averaged)")
    ax.set_title("per-channel reconstruction MSE  ·  green band = expected visual cortex")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"[phase0] device={DEVICE}  loading test set + checkpoints...")
    test = ThingsEEG(split="test")
    eeg_avg, _, _ = test.trial_averaged()        # (200, 63, 250) normalized
    print(f"[phase0] eeg_avg shape = {tuple(eeg_avg.shape)}")

    clf = load_holdout()

    # Pick the canonical "med" tier from v4 — bpps are comparable enough that
    # the localization story is robust to tier choice.
    codecs_for_recon = ["fidelity_v4_med", "neurozip_v4_med"]

    print("\n=========================================")
    print(" CHECK 1: per-channel ablation (held-out)")
    print("=========================================")
    raw_ablation     = check1_channel_ablation("raw",                 eeg_avg, clf, "raw EEG (no compression)")
    fid_ablation     = check1_channel_ablation("fidelity_v4_med",      eeg_avg, clf, "fidelity_v4_med")
    nzn_ablation     = check1_channel_ablation("neurozip_v4_med",      eeg_avg, clf, "neurozip_v4_med")

    print("\n=========================================")
    print(" CHECK 2: per-timepoint reconstruction MSE")
    print("=========================================")
    check2 = check2_per_timepoint_mse(codecs_for_recon, eeg_avg)
    # Sanity print: which timepoint window has highest fid-vs-nzn gap?
    fid_t = np.array(check2["fidelity_v4_med"]); nzn_t = np.array(check2["neurozip_v4_med"])
    gap_t = fid_t - nzn_t       # positive = NeuroZip lower MSE here
    t_ms = np.linspace(-200, 996, N_TIMES)
    for label_name, (a, b) in ERP_WINDOWS.items():
        mask = (t_ms >= a) & (t_ms <= b)
        rel_gap = float(gap_t[mask].mean() / fid_t[mask].mean() * 100)
        print(f"  {label_name:5s} window ({a}-{b} ms):  fid MSE = {fid_t[mask].mean():.5f}   "
              f"nzn MSE = {nzn_t[mask].mean():.5f}   relative gap = {rel_gap:+.1f}% "
              f"({'NeuroZip better' if rel_gap > 0 else 'fidelity better'})")

    print("\n=========================================")
    print(" CHECK 3: per-channel reconstruction MSE")
    print("=========================================")
    check3 = check3_per_channel_mse(codecs_for_recon, eeg_avg)
    fid_c = np.array(check3["fidelity_v4_med"]); nzn_c = np.array(check3["neurozip_v4_med"])
    visual_idx = [i for i, c in enumerate(CH_NAMES) if c in EXPECTED_VISUAL]
    other_idx  = [i for i in range(N_CHANNELS) if i not in visual_idx]
    print(f"  visual ch ({len(visual_idx)} of 63):  fid MSE = {fid_c[visual_idx].mean():.5f}   "
          f"nzn MSE = {nzn_c[visual_idx].mean():.5f}   "
          f"ratio nzn/fid = {nzn_c[visual_idx].mean()/fid_c[visual_idx].mean():.3f}")
    print(f"  other  ch ({len(other_idx)} of 63):  fid MSE = {fid_c[other_idx].mean():.5f}   "
          f"nzn MSE = {nzn_c[other_idx].mean():.5f}   "
          f"ratio nzn/fid = {nzn_c[other_idx].mean()/fid_c[other_idx].mean():.3f}")

    # ---------- write plots ----------
    plot_channel_ablation(raw_ablation, "raw EEG (no compression)",
                          OUT_DIR / "phase0_ablation_raw.png")
    plot_channel_ablation(fid_ablation, "fidelity_v4_med",
                          OUT_DIR / "phase0_ablation_fidelity.png")
    plot_channel_ablation(nzn_ablation, "neurozip_v4_med",
                          OUT_DIR / "phase0_ablation_neurozip.png")
    plot_per_timepoint(check2, OUT_DIR / "phase0_timepoint_mse.png")
    plot_per_channel_mse(check3, OUT_DIR / "phase0_channel_mse.png")

    summary = {
        "device": DEVICE,
        "ch_names": CH_NAMES,
        "expected_visual": sorted(EXPECTED_VISUAL),
        "erp_windows_ms": ERP_WINDOWS,
        "check1_ablation": {"raw": raw_ablation, "fidelity_v4_med": fid_ablation, "neurozip_v4_med": nzn_ablation},
        "check2_per_timepoint_mse": check2,
        "check3_per_channel_mse": check3,
    }
    (OUT_DIR / "phase0_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- verdict ----------
    print("\n=========================================")
    print(" VERDICT")
    print("=========================================")
    # Spatial story (Check 1 + 3)
    n_50 = nzn_ablation["channels_for_50pct_of_drop"]
    n_visual_in_top15 = nzn_ablation["n_visual_in_top15"]
    visual_ratio = nzn_c[visual_idx].mean() / (nzn_c.mean() + 1e-9)
    print(f"  SPATIAL:")
    print(f"    Channels needed for 50% of ablation drop (NeuroZip): {n_50}/63")
    print(f"    Visual-cortex channels in top-15: {n_visual_in_top15}/15  (target ≥6)")
    print(f"    Visual ch MSE / mean ch MSE: {visual_ratio:.3f}  (≪1 = NeuroZip preserves visual better)")
    spatial_ok = (n_50 <= 25) and (n_visual_in_top15 >= 6)
    print(f"    → SPATIAL story holds: {'YES' if spatial_ok else 'WEAK'}")
    # Temporal story (Check 2)
    n170_mask = (t_ms >= ERP_WINDOWS["N170"][0]) & (t_ms <= ERP_WINDOWS["N170"][1])
    p300_mask = (t_ms >= ERP_WINDOWS["P300"][0]) & (t_ms <= ERP_WINDOWS["P300"][1])
    n170_relgap = (fid_t[n170_mask].mean() - nzn_t[n170_mask].mean()) / fid_t[n170_mask].mean() * 100
    p300_relgap = (fid_t[p300_mask].mean() - nzn_t[p300_mask].mean()) / fid_t[p300_mask].mean() * 100
    print(f"  TEMPORAL:")
    print(f"    N170 (150-200 ms) NeuroZip-vs-fidelity MSE gap: {n170_relgap:+.1f}% (positive = NeuroZip preserves better)")
    print(f"    P300 (280-400 ms) NeuroZip-vs-fidelity MSE gap: {p300_relgap:+.1f}%")
    temporal_ok = (n170_relgap > 5) or (p300_relgap > 5)
    print(f"    → TEMPORAL story holds: {'YES' if temporal_ok else 'WEAK'}")
    # Held-out under aggressive ablation
    held_robust = nzn_ablation["base_acc"] > 0.9
    print(f"  ROBUSTNESS:")
    print(f"    Held-out classifier base accuracy on NeuroZip recon: {nzn_ablation['base_acc']*100:.1f}%")
    print(f"    → '100% accuracy survives 144× compression' headline: {'TRUE' if held_robust else 'FALSE'}")
    print()
    n_pass = sum([spatial_ok, temporal_ok, held_robust])
    print(f"  OVERALL: {n_pass}/3 checks support the biological reframe.")
    if n_pass >= 2:
        print("  → GREEN: proceed to Phase 1 (biology figures + reframe).")
    elif n_pass == 1:
        print("  → YELLOW: partial story; tighten claims accordingly.")
    else:
        print("  → RED: reframe would be rhetoric. Stick with technical-merit pitch.")


if __name__ == "__main__":
    main()
