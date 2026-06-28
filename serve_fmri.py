"""
serve_fmri.py — port-8011 viewer for the fMRI v4 codec.

For a held-out fMRI window (200 ROIs x 64 TRs): the ORIGINAL ("before latent"),
the ENCODED latent, the RECONSTRUCTION, the DIFFERENCE (overlay subtraction), a
few ROI time-course overlays, and stats (MSE, variance explained, original vs
compressed size, ratio). Continuous BOLD -> compares vs float16 (legitimate here,
fMRI is stored as floats).

Run:  ./serve_fmri.sh   # or python serve_fmri.py --port 8011
"""
import argparse, io, os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response

from fmri_data import load_fmri
from fmri_codec import fMRICodec, N_ROI, N_TIMES

app = Flask(__name__)
S = {}
SRC_BITS = N_ROI * N_TIMES * 16

CSS = """
:root{--bg:#f6f7fa;--surface:#fff;--border:#e3e6ee;--ink:#0f1322;--ink-2:#4a5163;
--ink-muted:#8a92a8;--accent:#7c3aed;--shadow:0 1px 2px rgba(15,19,34,.04),0 8px 24px rgba(15,19,34,.05);--radius:14px}
*{box-sizing:border-box}body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:0 16px 60px}
.wrap{max-width:1100px;margin:0 auto}header{display:flex;align-items:baseline;gap:14px;padding:24px 0 4px}
h1{margin:0;font-size:1.7em;letter-spacing:-.5px}h1 span{color:var(--accent)}.sub{color:var(--ink-2);margin:2px 0 0}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
.cell{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:11px 16px;box-shadow:var(--shadow);min-width:120px}
.cell .v{font-size:1.35em;font-weight:700}.cell .v.us{color:var(--accent)}.cell .k{color:var(--ink-muted);font-size:.74em;text-transform:uppercase;letter-spacing:.5px}
.controls{display:flex;gap:18px;align-items:center;margin:6px 0 14px}select,button{font-size:15px;padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:var(--surface)}button{cursor:pointer;font-weight:600}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px;box-shadow:var(--shadow);margin-top:16px}
.card h3{margin:2px 0 8px;font-size:.92em;color:var(--ink-2)}.card img{width:100%;border-radius:10px;display:block}
.note{color:var(--ink-muted);font-size:.85em;margin-top:16px}
"""


def load_codec(ckpt, device):
    ck = torch.load(ckpt, weights_only=False, map_location=device)
    c = ck["config"]
    m = fMRICodec(n_roi=c["n_roi"], c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck


def _compute(i):
    m = S["model"]; x = S["va"][i:i+1]
    with torch.no_grad():
        y = torch.round(m.encoder(x))
        xh = m.decoder(y)
        bps = float(m.prior.bits(y).cpu())
    comp_bits = bps * m.c_lat * 8
    xo = x[0].cpu().numpy(); xr = xh[0].cpu().numpy()
    mse = float(((xo - xr) ** 2).mean())
    return {"xo": xo, "xr": xr, "lat": y[0].cpu().numpy(), "mse": mse, "comp_bits": comp_bits}


@app.route("/api/info")
def api_info():
    i = int(request.args.get("idx", 0)); d = _compute(i)
    var = float(S["va"][i].var().cpu())
    comp_bytes = d["comp_bits"] / 8
    return jsonify({"mse": d["mse"], "var_exp": 1 - d["mse"] / var,
                    "orig_bytes": SRC_BITS / 8, "comp_bytes": comp_bytes,
                    "ratio": SRC_BITS / d["comp_bits"]})


def _png(fig):
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=108); plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")


@app.route("/api/panels")
def panels():
    i = int(request.args.get("idx", 0)); d = _compute(i)
    xo, xr, diff = d["xo"], d["xr"], d["xo"] - d["xr"]
    fig, ax = plt.subplots(4, 1, figsize=(11, 8.4), gridspec_kw={"hspace": 0.55})
    vmax = float(np.abs(np.concatenate([xo, xr])).max())
    ax[0].imshow(xo, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[0].set_title("BEFORE latent — original fMRI (200 ROIs × 64 TRs)", fontsize=10, loc="left")
    ax[1].imshow(d["lat"], aspect="auto", cmap="PuOr", interpolation="nearest")
    ax[1].set_title(f"ENCODED latent — {d['lat'].shape[0]}×{d['lat'].shape[1]} integer codes (compressed form)", fontsize=10, loc="left")
    ax[2].imshow(xr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[2].set_title(f"RECONSTRUCTION  (MSE {d['mse']:.3f})", fontsize=10, loc="left")
    dm = float(np.abs(diff).max()) or 1.0
    ax[3].imshow(diff, aspect="auto", cmap="seismic", vmin=-dm, vmax=dm)
    ax[3].set_title("DIFFERENCE — original − reconstruction", fontsize=10, loc="left")
    for a in (ax[0], ax[2], ax[3]):
        a.set_ylabel("ROI"); a.set_xlabel("TR")
    ax[1].set_xlabel("latent time")
    return _png(fig)


@app.route("/api/traces")
def traces():
    i = int(request.args.get("idx", 0)); d = _compute(i)
    rois = [10, 60, 110, 160]
    t = np.arange(N_TIMES) * 2
    fig, ax = plt.subplots(len(rois), 1, figsize=(11, 5), sharex=True)
    for a, r in zip(ax, rois):
        a.plot(t, d["xo"][r], color="0.4", lw=1.2, label="original")
        a.plot(t, d["xr"][r], color="#7c3aed", lw=1.2, alpha=0.85, label="reconstruction")
        a.set_ylabel(f"ROI {r}", fontsize=8)
    ax[0].legend(fontsize=8, loc="upper right"); ax[-1].set_xlabel("time (s)")
    fig.suptitle("BOLD time-courses — original vs reconstruction", fontsize=11)
    return _png(fig)


@app.route("/")
def index():
    opts = "".join(f'<option value="{i}">window {i:03d}</option>' for i in range(S["va"].shape[0]))
    fv = S["meta"]["final_val"]
    return f"""<!doctype html><html><head><meta charset=utf-8><title>NeuroZip · fMRI codec</title>
<style>{CSS}</style></head><body><div class=wrap>
<header><h1>Neuro<span>Zip</span></h1><div class=sub>v4 codec on fMRI — same architecture as EEG (63→200 ROIs) · ABIDE CC200 parcellated BOLD</div></header>
<div class=stats id=stats></div>
<div class=controls><label>window&nbsp;<select id=sel onchange=upd()>{opts}</select></label><button onclick=rnd()>Random</button></div>
<div class=card><h3>original · latent · reconstruction · difference</h3><img id=panels></div>
<div class=card><h3>example ROI BOLD time-courses (original vs reconstruction)</h3><img id=traces></div>
<p class=note>The raw fMRI volume is 3-D+time (huge); the right 1-D form is the parcellated ROI×time series.
BOLD is slow/smooth, so the same conv codec compresses it well. Headline: {fv['ratio']:.0f}× at {fv['var_exp']*100:.0f}% variance explained.</p>
<script>
async function info(i){{const d=await (await fetch('/api/info?idx='+i)).json();
 document.getElementById('stats').innerHTML=
  `<div class=cell><div class=v>${{(d.orig_bytes/1000).toFixed(1)}} KB</div><div class=k>original (float16)</div></div>`+
  `<div class=cell><div class="v us">${{(d.comp_bytes/1000).toFixed(2)}} KB</div><div class=k>compressed</div></div>`+
  `<div class=cell><div class="v us">${{d.ratio.toFixed(0)}}×</div><div class=k>smaller</div></div>`+
  `<div class=cell><div class="v us">${{(d.var_exp*100).toFixed(0)}}%</div><div class=k>variance explained</div></div>`+
  `<div class=cell><div class=v>${{d.mse.toFixed(4)}}</div><div class=k>recon MSE</div></div>`;}}
function upd(){{const i=document.getElementById('sel').value,t=Date.now();
 document.getElementById('panels').src='/api/panels?idx='+i+'&t='+t;
 document.getElementById('traces').src='/api/traces?idx='+i+'&t='+t; info(i);}}
function rnd(){{const s=document.getElementById('sel');s.value=Math.floor(Math.random()*s.options.length);upd();}}
upd();
</script></div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/fmri_codec.pt")
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    model, ck = load_codec(args.ckpt, device)
    _, val, _ = load_fmri()
    S.update(model=model, va=torch.from_numpy(val).to(device), meta=ck)
    print(f"serving fMRI viewer on http://{args.host}:{args.port}/  "
          f"(ratio {ck['final_val']['ratio']:.0f}x, var {ck['final_val']['var_exp']*100:.0f}%)")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
