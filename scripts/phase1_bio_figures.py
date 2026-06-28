"""Phase 1 — biology figures + headline numbers for the reframed pitch.

Three slide-ready artifacts derived from already-computed v4 checkpoints.
Run AFTER phase0_localization.py (uses plots/phase0_summary.json if
present; recomputes if missing).

The tier matters for honest reporting: bio numbers and the compression
ratio quoted in the pitch must come from the same run. Default is `low`
(matches the 144× headline). Pass `--tier med` or `--tier high` to
recompute against another operating point and overwrite the artifacts.

Outputs to plots/:
  phase1_erp_timeline.png   — hero: per-timepoint MSE with ERP windows
  phase1_topographic.png    — real 10-20 topographic map of NeuroZip's
                              MSE improvement over fidelity per channel
  phase1_summary.png        — three-panel composite for the "money slide"
  phase1_bio_numbers.json   — headline numbers for results.md / pitch
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import ThingsEEG, N_CHANNELS, N_TIMES
from clip_proj import load_frozen_projector
from codec import EEGCodec, CHECKPOINTS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CH_NAMES = ['Fp1','Fp2','AF7','AF3','AFz','AF4','AF8','F7','F5','F3','F1','Fz',
            'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6',
            'FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3',
            'CP1','CPz','CP2','CP4','CP6','TP8','P9','P7','P5','P3','P1','Pz','P2',
            'P4','P6','P8','P10','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','Iz']

# Visual / object-recognition cortex.
EXPECTED_VISUAL = {'O1','O2','Oz','Iz','PO7','PO8','PO3','PO4','POz','P7','P8','P9','P10'}

ERP_WINDOWS = {
    "P100": (90, 130),
    "N170": (150, 200),
    "P200": (200, 260),
    "P300": (280, 400),
}

# Pitch palette
RED  = "#d62728"      # NeuroZip
GRAY = "#7c8497"      # Fidelity
INK  = "#161a22"
MUTE = "#8b92a4"


# ---------------------------------------------------------------------------
# Loaders + recompute fallback
# ---------------------------------------------------------------------------
def load_codec(name: str) -> EEGCodec:
    path = CHECKPOINTS / f"{name}.pt"
    state = torch.load(path, weights_only=False, map_location=DEVICE)
    cfg = state.get("config", {})
    m = EEGCodec(c_lat=cfg.get("c_lat", 32), hidden=cfg.get("hidden", 128),
                  n_attn=cfg.get("n_attn", 0)).to(DEVICE)
    m.load_state_dict(state["model"]); m.eval()
    return m


@torch.no_grad()
def per_channel_per_time_mse(codec_name: str, eeg_avg: torch.Tensor):
    """Returns (mse_per_ch [63], mse_per_t [250]) for the given codec."""
    c = load_codec(codec_name)
    hats = []
    for i in range(0, eeg_avg.size(0), 200):
        xh, _ = c.compress_then_reconstruct(eeg_avg[i:i+200].to(DEVICE))
        hats.append(xh.cpu())
    recon = torch.cat(hats, dim=0)
    err = (eeg_avg - recon).pow(2)
    return err.mean(dim=(0, 2)).numpy(), err.mean(dim=(0, 1)).numpy()


# ---------------------------------------------------------------------------
# Figure 1: ERP timeline (hero plot — clean and big)
# ---------------------------------------------------------------------------
def fig_erp_timeline(fid_t: np.ndarray, nzn_t: np.ndarray, save_to: Path):
    t_ms = np.linspace(-200, 996, N_TIMES)
    fig, ax = plt.subplots(figsize=(11, 4.4))

    for label_name, (a, b) in ERP_WINDOWS.items():
        ax.axvspan(a, b, alpha=0.12, color="#1f9d55", zorder=0)
        ax.text((a + b) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0,
                label_name, ha="center", va="bottom",
                fontsize=10, color="#1f9d55", fontweight="bold")

    ax.plot(t_ms, fid_t, color=GRAY, lw=1.8, label="fidelity codec (MSE-only)", alpha=0.9)
    ax.plot(t_ms, nzn_t, color=RED, lw=2.0, label="NeuroZip (task-aware)")
    ax.fill_between(t_ms, nzn_t, fid_t, where=(fid_t > nzn_t),
                     color=RED, alpha=0.08, label="NeuroZip preserves better")

    ax.axvline(0, color=INK, lw=0.8, alpha=0.6)
    ax.text(2, ax.get_ylim()[1] * 0.95, "stimulus onset",
            ha="left", va="top", fontsize=9, color=INK, alpha=0.7)

    # Re-place ERP labels with the actual final ylim
    ax.set_ylim(bottom=0)
    ymax = ax.get_ylim()[1]
    for child in list(ax.get_children()):
        if isinstance(child, plt.Text) and child.get_text() in ERP_WINDOWS:
            child.set_y(ymax * 0.985)

    ax.set_xlabel("time after stimulus (ms)", fontsize=11)
    ax.set_ylabel("reconstruction MSE  (channel- and epoch-averaged)", fontsize=11)
    ax.set_title("Where the meaning lives: NeuroZip preserves the visual-evoked ERP windows tighter",
                 fontsize=12, color=INK, loc="left", pad=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Headline annotation: white opaque background so the text reads
    # cleanly even when it sits over the ERP windows and data lines.
    fid_n170 = fid_t[(t_ms >= 150) & (t_ms <= 200)].mean()
    nzn_n170 = nzn_t[(t_ms >= 150) & (t_ms <= 200)].mean()
    pct = (fid_n170 - nzn_n170) / fid_n170 * 100
    # Park the callout in the quiet post-400 ms region where neither curve
    # is moving much, so the box doesn't sit on top of the action.
    ax.annotate(f"N170 window:\nNeuroZip preserves it\n{pct:.1f}% better than fidelity",
                xy=(175, nzn_n170), xytext=(520, ax.get_ylim()[1] * 0.62),
                fontsize=10.5, color="#003366", fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.55",
                          facecolor="white", edgecolor="#003366",
                          linewidth=1.0, alpha=0.96),
                arrowprops=dict(arrowstyle="-|>", color="#003366",
                                lw=1.4, connectionstyle="arc3,rad=-0.18"))
    fig.tight_layout()
    fig.savefig(save_to, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: topographic map (real 10-20 layout via MNE)
# ---------------------------------------------------------------------------
def fig_topographic(fid_ch: np.ndarray, nzn_ch: np.ndarray, save_to: Path):
    """Plot 3 panels: fidelity per-channel MSE, NeuroZip per-channel MSE,
    and the improvement (fidelity - NeuroZip)."""
    info = mne.create_info(ch_names=CH_NAMES, sfreq=250.0, ch_types="eeg")
    info.set_montage("standard_1020", on_missing="ignore")

    # Diff = how much better NeuroZip is per channel (positive = NeuroZip wins)
    diff = fid_ch - nzn_ch
    fid_n170_ratio = (nzn_ch / np.maximum(fid_ch, 1e-9))  # <1 means NeuroZip better

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8),
                             gridspec_kw={"width_ratios":[1,1,1.06]})

    vmax = float(max(fid_ch.max(), nzn_ch.max()))
    im_f, _ = mne.viz.plot_topomap(fid_ch, info, axes=axes[0], cmap="Reds",
                                     vlim=(0, vmax), show=False, sensors=True,
                                     contours=4)
    axes[0].set_title("Fidelity\nreconstruction MSE per channel", fontsize=11, color=GRAY, fontweight="bold")
    im_n, _ = mne.viz.plot_topomap(nzn_ch, info, axes=axes[1], cmap="Reds",
                                     vlim=(0, vmax), show=False, sensors=True,
                                     contours=4)
    axes[1].set_title("NeuroZip\nreconstruction MSE per channel", fontsize=11, color=RED, fontweight="bold")

    vmax_d = float(max(diff.max(), -diff.min()))
    im_d, _ = mne.viz.plot_topomap(diff, info, axes=axes[2], cmap="RdBu_r",
                                     vlim=(-vmax_d, vmax_d), show=False, sensors=True,
                                     contours=4)
    axes[2].set_title("NeuroZip advantage\n(fidelity − NeuroZip MSE)", fontsize=11, color="#003366", fontweight="bold")

    # Shared colorbar for the first two
    cax = fig.add_axes([0.005, 0.18, 0.012, 0.62])
    cb = plt.colorbar(im_f, cax=cax)
    cb.set_label("MSE (σ²)", fontsize=9, color=INK)
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.set_label_position("left"); cb.ax.yaxis.tick_left()
    # Separate cbar for diff panel
    cax2 = fig.add_axes([0.95, 0.18, 0.012, 0.62])
    cb2 = plt.colorbar(im_d, cax=cax2)
    cb2.set_label("MSE difference", fontsize=9, color=INK)
    cb2.ax.tick_params(labelsize=8)

    fig.suptitle("Object-discriminative information is concentrated in visual cortex — and NeuroZip preserves it tighter",
                 fontsize=12, color=INK, y=1.02, fontweight="normal")
    fig.subplots_adjust(left=0.05, right=0.93, top=0.85, bottom=0.05)
    fig.savefig(save_to, dpi=140, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: 3-panel composite "money slide"
# ---------------------------------------------------------------------------
def fig_summary_composite(fid_t, nzn_t, fid_ch, nzn_ch, save_to: Path, nzn_ratio: int = 144):
    t_ms = np.linspace(-200, 996, N_TIMES)

    fig = plt.figure(figsize=(14, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.05], hspace=0.4, wspace=0.15)

    # Panel 1: ERP timeline
    ax1 = fig.add_subplot(gs[0, 0])
    for _, (a, b) in ERP_WINDOWS.items():
        ax1.axvspan(a, b, alpha=0.12, color="#1f9d55", zorder=0)
    ax1.plot(t_ms, fid_t, color=GRAY, lw=1.3, label="fidelity")
    ax1.plot(t_ms, nzn_t, color=RED,  lw=1.6, label="NeuroZip")
    ax1.axvline(0, color=INK, lw=0.7, alpha=.5)
    ax1.set_xlabel("ms after stimulus", fontsize=9)
    ax1.set_ylabel("MSE", fontsize=9)
    ax1.set_title("WHEN: ERP windows preserved tighter under NeuroZip",
                  fontsize=10, color=INK, loc="left", pad=6, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")

    # Panel 2: per-channel bar (visual cortex highlighted), grouped by ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ratio = nzn_ch / np.maximum(fid_ch, 1e-9)
    sort_idx = np.argsort(ratio)         # lowest ratio = NeuroZip wins most
    colors = ['#1f9d55' if CH_NAMES[i] in EXPECTED_VISUAL else GRAY for i in sort_idx]
    ax2.bar(range(N_CHANNELS), ratio[sort_idx], color=colors, width=0.85)
    ax2.axhline(1.0, color=INK, lw=0.8, linestyle="--", alpha=.5)
    ax2.text(1, 1.02, "ratio = 1 (no advantage)", fontsize=8, color=INK, alpha=.6)
    ax2.set_xticks([])
    ax2.set_xlabel(f"channels ranked by NeuroZip advantage  ({len(EXPECTED_VISUAL)} visual cortex in green)",
                   fontsize=9)
    ax2.set_ylabel("NeuroZip / fidelity MSE", fontsize=9)
    ax2.set_title("WHERE it lives: visual-cortex channels preserved most",
                  fontsize=10, color=INK, loc="left", pad=6, fontweight="bold")
    ax2.set_ylim(0.4, max(1.2, ratio.max() * 1.05))

    # Panel 3: text summary card
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    # Compute the displayed numbers from the data we just made, so this
    # text never falls out of sync with the JSON when the tier changes.
    visual_idx = [i for i, c in enumerate(CH_NAMES) if c in EXPECTED_VISUAL]
    other_idx  = [i for i in range(N_CHANNELS) if i not in visual_idx]
    vis_pct = (1 - nzn_ch[visual_idx].mean() / fid_ch[visual_idx].mean()) * 100
    oth_pct = (1 - nzn_ch[other_idx].mean()  / fid_ch[other_idx].mean()) * 100
    pref    = vis_pct / max(oth_pct, 1e-9)
    t_ms = np.linspace(-200, 996, N_TIMES)
    erp = lambda a, b: ((fid_t[(t_ms>=a)&(t_ms<=b)].mean() - nzn_t[(t_ms>=a)&(t_ms<=b)].mean())
                        / fid_t[(t_ms>=a)&(t_ms<=b)].mean() * 100)
    summary_text = (
        f"  At {nzn_ratio}× compression, NeuroZip preserves\n"
        f"  visual-cortex EEG signal far tighter than\n"
        f"  fidelity at matched architecture.\n\n"
        "  WHERE the gap concentrates:\n"
        f"     visual cortex MSE: {vis_pct:.1f}% below fidelity\n"
        f"     other channels:    {oth_pct:.1f}% below fidelity\n"
        f"     spatial preference: {pref:.1f}× larger\n\n"
        "  WHEN it concentrates:\n"
        f"     N170 (150–200 ms): {erp(150,200):.1f}% tighter\n"
        f"     P200 (200–260 ms): {erp(200,260):.1f}% tighter\n"
        f"     P100 (90–130 ms):  {erp(90,130):.1f}% tighter\n"
        f"     P300 (280–400 ms): {erp(280,400):.1f}% tighter\n"
    )
    ax3.text(0.02, 0.98, "THE BIOLOGY", fontsize=11, color=RED, fontweight="bold",
              va="top", transform=ax3.transAxes)
    ax3.text(0.02, 0.88, summary_text, fontsize=10, color=INK, va="top",
              transform=ax3.transAxes, family="monospace", linespacing=1.45)

    fig.suptitle(f"NeuroZip preserves visual-cortex EEG signal tighter at {nzn_ratio}× compression — exactly where neuroscience predicts",
                 fontsize=13, color=INK, y=0.99, fontweight="normal")
    fig.savefig(save_to, dpi=140, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TIER_META = {
    # tier → (fid_name, nzn_name, neurozip bpp, neurozip ratio)
    "low":   ("fidelity_v4_low",   "neurozip_v4_low",   0.111, 144),
    "med":   ("fidelity_v4_med",   "neurozip_v4_med",   0.190,  84),
    "high":  ("fidelity_v4_high",  "neurozip_v4_high",  0.222,  72),
    "xhigh": ("fidelity_v4_xhigh", "neurozip_v4_xhigh", 0.250,  64),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=list(TIER_META), default="low",
                   help="Which v4 operating point to analyze. Default low "
                        "(144×) so bio numbers align with the pitch's "
                        "headline compression ratio.")
    args = p.parse_args()
    fid_name, nzn_name, nzn_bpp, nzn_ratio = TIER_META[args.tier]
    print(f"[phase1] tier={args.tier} → {nzn_name} ({nzn_ratio}×, {nzn_bpp} bpp)")

    test = ThingsEEG(split="test")
    eeg_avg, _, _ = test.trial_averaged()                # (200, 63, 250)

    print("[phase1] computing per-channel + per-timepoint MSE…")
    fid_ch, fid_t = per_channel_per_time_mse(fid_name, eeg_avg)
    nzn_ch, nzn_t = per_channel_per_time_mse(nzn_name, eeg_avg)

    # Numerical summary
    visual_idx = [i for i, c in enumerate(CH_NAMES) if c in EXPECTED_VISUAL]
    other_idx  = [i for i in range(N_CHANNELS) if i not in visual_idx]
    visual_ratio = float(nzn_ch[visual_idx].mean() / fid_ch[visual_idx].mean())
    other_ratio  = float(nzn_ch[other_idx].mean()  / fid_ch[other_idx].mean())
    t_ms = np.linspace(-200, 996, N_TIMES)
    erp_gaps = {
        n: float((fid_t[(t_ms >= a) & (t_ms <= b)].mean()
                  - nzn_t[(t_ms >= a) & (t_ms <= b)].mean())
                 / fid_t[(t_ms >= a) & (t_ms <= b)].mean() * 100)
        for n, (a, b) in ERP_WINDOWS.items()
    }
    headline = {
        "tier": args.tier,
        "fid_codec": fid_name,
        "nzn_codec": nzn_name,
        "spatial": {
            "visual_cortex_nzn_to_fid_mse_ratio": visual_ratio,
            "other_channels_nzn_to_fid_mse_ratio": other_ratio,
            "spatial_preference_x_larger_in_visual": (1 - visual_ratio) / max(1 - other_ratio, 1e-9),
            "visual_pct_better": (1 - visual_ratio) * 100,
            "other_pct_better": (1 - other_ratio) * 100,
        },
        "temporal_erp_pct_tighter": erp_gaps,
        "compression": {
            "neurozip_bpp": nzn_bpp,
            "neurozip_compression_ratio": nzn_ratio,
        },
        # Cached per-channel arrays so phase2_permutation can reuse them
        # without re-running the codec forward passes.
        "_per_channel_mse": {
            "fidelity": fid_ch.tolist(),
            "neurozip": nzn_ch.tolist(),
        },
    }
    (OUT / "phase1_bio_numbers.json").write_text(json.dumps(headline, indent=2))

    print("[phase1] headline numbers:")
    print(f"  visual cortex: NeuroZip MSE is {(1-visual_ratio)*100:.1f}% below fidelity")
    print(f"  other channels: {(1-other_ratio)*100:.1f}% below fidelity")
    print(f"  -> spatial preference {((1-visual_ratio)/(1-other_ratio)):.1f}x stronger in visual cortex")
    for n, g in erp_gaps.items():
        print(f"  {n}: NeuroZip MSE {g:+.1f}% vs fidelity")

    print("[phase1] writing figures…")
    fig_erp_timeline(fid_t, nzn_t, OUT / "phase1_erp_timeline.png")
    fig_topographic(fid_ch, nzn_ch, OUT / "phase1_topographic.png")
    fig_summary_composite(fid_t, nzn_t, fid_ch, nzn_ch, OUT / "phase1_summary.png",
                          nzn_ratio=nzn_ratio)
    print(f"[phase1] done. plots written to {OUT}/phase1_*.png")


if __name__ == "__main__":
    main()
