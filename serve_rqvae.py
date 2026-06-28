"""
serve_rqvae.py — port-8011 viewer for the RQ-VAE codec.

Polished reconstruction viewer (UI styled after rian's demo_clean.html): pick a
held-out concept, see the stimulus image the subject saw alongside the trial-
averaged ACTUAL EEG, the RQ-VAE RECONSTRUCTION, the error, the discrete latent
codes, and per-channel waveforms. Self-contained (no open_clip / judge needed).

Run:
    ./serve_rqvae.sh                                  # or:
    python serve_rqvae.py --port 8011 --ckpt checkpoints/codec_rqvae_72x.pt
    # then open http://localhost:8011/
"""

import argparse
import io
import os
import urllib.request
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response

from data import ThingsEEG, N_TIMES
from rqvae import EEGCodecRQ

app = Flask(__name__)
S = {}
HF_IMG = "https://huggingface.co/datasets/Haitao999/things-eeg/resolve/main/Image_set/"
STIM_CACHE = Path("results/stimuli"); STIM_CACHE.mkdir(parents=True, exist_ok=True)

CSS = """
:root{--bg:#f6f7fa;--surface:#fff;--surface-2:#f0f2f7;--border:#e3e6ee;--ink:#0f1322;
--ink-2:#4a5163;--ink-muted:#8a92a8;--accent:#d62728;--ok:#1f9d55;
--shadow:0 1px 2px rgba(15,19,34,.04),0 8px 24px rgba(15,19,34,.05);--radius:14px}
*{box-sizing:border-box}body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,sans-serif;
background:var(--bg);color:var(--ink);margin:0;padding:0 16px 60px}
.wrap{max-width:1080px;margin:0 auto}
header{display:flex;align-items:baseline;gap:14px;padding:28px 0 6px}
h1{margin:0;font-size:1.7em;letter-spacing:-.5px}h1 span{color:var(--accent)}
.sub{color:var(--ink-2);margin:2px 0 0}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}
.cell{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
padding:12px 18px;box-shadow:var(--shadow);min-width:120px}
.cell .v{font-size:1.45em;font-weight:700}.cell .v.us{color:var(--accent)}
.cell .k{color:var(--ink-muted);font-size:.78em;text-transform:uppercase;letter-spacing:.5px}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:8px 0 16px}
select,button{font-size:15px;padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:var(--surface)}
button{cursor:pointer;font-weight:600}button:hover{background:var(--surface-2)}
.btn-go{background:var(--accent);color:#fff;border-color:var(--accent)}
.row{display:grid;grid-template-columns:240px 1fr;gap:18px;align-items:start}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
padding:12px;box-shadow:var(--shadow);margin-top:16px}
.card h3{margin:2px 0 8px;font-size:.95em;color:var(--ink-2)}
.card img,.stim img{width:100%;border-radius:10px;display:block}
.stim{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px;box-shadow:var(--shadow)}
.stim .cap{color:var(--ink-muted);font-size:.82em;margin-top:6px;text-align:center}
.note{color:var(--ink-muted);font-size:.85em;margin-top:18px}
"""


def load(ckpt, device):
    ck = torch.load(ckpt, weights_only=False, map_location=device)
    c = ck["config"]
    m = EEGCodecRQ(c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"],
                   codebook_size=c["codebook_size"], num_quantizers=c["num_quantizers"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck


@app.route("/")
def index():
    opts = "".join(f'<option value="{i}">{i:03d} — {t}</option>' for i, t in enumerate(S["concepts"]))
    fv = S["meta"]["final_val"]; cfg = S["meta"]["config"]
    return f"""<!doctype html><html><head><meta charset=utf-8><title>NeuroZip · RQ-VAE</title>
<style>{CSS}</style></head><body><div class=wrap>
<header><h1>Neuro<span>Zip</span></h1><div class=sub>RQ-VAE codec — residual vector-quantized latent · held-out, trial-averaged test EEG</div></header>
<div class=stats>
 <div class=cell><div class=v us>{S['ratio']:.0f}×</div><div class=k>compression vs fp16</div></div>
 <div class=cell><div class=v>{fv['mse']:.4f}</div><div class=k>test-avg MSE</div></div>
 <div class=cell><div class=v>{fv.get('var_exp',0)*100:.0f}%</div><div class=k>variance explained</div></div>
 <div class=cell><div class=v>D={cfg['num_quantizers']}·K={cfg['codebook_size']}</div><div class=k>residual codebooks</div></div>
 <div class=cell><div class=v>{cfg['c_lat']}×32</div><div class=k>latent tokens×dim</div></div>
 <div class=cell><div class=v>{cfg['num_quantizers']*32}</div><div class=k>codes / epoch</div></div>
</div>
<div class=controls>
 <label>concept&nbsp;<select id=sel onchange=upd()>{opts}</select></label>
 <button onclick=rnd()>Random</button>
</div>
<div class=row>
 <div class=stim><img id=stim><div class=cap id=stimcap>stimulus</div></div>
 <div class=card><h3>actual · RQ-VAE reconstruction · error (normalized σ)</h3><img id=heat></div>
</div>
<div class=card><h3>discrete latent codes (summed residual quantizers) · per-channel waveforms</h3><img id=detail></div>
<p class=note>The error panel is high-frequency grain — the RQ-VAE keeps the structured signal and
discards single-trial noise. Latent = {cfg['num_quantizers']*32} discrete codes/epoch
({cfg['num_quantizers']} residual codebooks × 32 tokens). No live model inference happens in the browser.</p>
<script>
function upd(){{const i=document.getElementById('sel').value,t=Date.now();
document.getElementById('heat').src='/api/heat?idx='+i+'&t='+t;
document.getElementById('detail').src='/api/detail?idx='+i+'&t='+t;
document.getElementById('stim').src='/api/stimulus?idx='+i+'&t='+t;
document.getElementById('stimcap').textContent=document.getElementById('sel').selectedOptions[0].textContent.split('— ')[1];}}
function rnd(){{const s=document.getElementById('sel');s.value=Math.floor(Math.random()*s.options.length);upd();}}
upd();
</script></div></body></html>"""


def _recon(i):
    x = S["te"][i:i+1]
    with torch.no_grad():
        xh, _ = S["model"].compress_then_reconstruct(x)
        z = S["model"].encoder(x).transpose(1, 2)
        zq, codes, _ = S["model"].rvq(z)
    return x[0].cpu().numpy(), xh[0].cpu().numpy(), codes[0].cpu().numpy()


def _png(fig):
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=108); plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")


@app.route("/api/heat")
def heat():
    i = int(request.args.get("idx", 0))
    xo, xr, _ = _recon(i)
    diff = xo - xr; mse = float((diff ** 2).mean())
    vmax = float(np.abs(np.concatenate([xo, xr])).max())
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    ext = [-200, 996, 63, 0]
    ax[0].imshow(xo, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext); ax[0].set_title(f"ACTUAL — {S['concepts'][i]}")
    ax[1].imshow(xr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext); ax[1].set_title("RQ-VAE RECONSTRUCTION")
    dm = float(np.abs(diff).max()) or 1.0
    ax[2].imshow(diff, aspect="auto", cmap="seismic", vmin=-dm, vmax=dm, extent=ext); ax[2].set_title(f"ERROR  MSE={mse:.4f}")
    for a in ax: a.set_xlabel("time after stim (ms)"); a.set_ylabel("channel")
    return _png(fig)


@app.route("/api/detail")
def detail():
    i = int(request.args.get("idx", 0))
    xo, xr, codes = _recon(i)            # codes: (32 tokens, D)
    fig = plt.figure(figsize=(12, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.6, 1.6], wspace=0.32)
    axc = fig.add_subplot(gs[0])
    axc.imshow(codes.T, aspect="auto", cmap="viridis", interpolation="nearest")
    axc.set_title(f"latent codes ({codes.shape[1]}×32)"); axc.set_xlabel("token (time)"); axc.set_ylabel("residual quantizer")
    t = np.linspace(-200, 996, N_TIMES)
    for k, ch in enumerate([18, 47]):
        ax = fig.add_subplot(gs[k + 1])
        ax.plot(t, xo[ch], color="0.35", lw=1.3, label="actual")
        ax.plot(t, xr[ch], color="#d62728", lw=1.1, label="reconstruction")
        ax.axvline(0, color="0.8", lw=.7); ax.set_title(f"channel {ch}"); ax.set_xlabel("time after stim (ms)")
        if k == 0: ax.legend(fontsize=8)
    return _png(fig)


@app.route("/api/stimulus")
def stimulus():
    i = int(request.args.get("idx", 0))
    path = S["img_paths"][i]
    local = STIM_CACHE / os.path.basename(path)
    if not local.exists():
        try:
            req = urllib.request.Request(HF_IMG + path, headers={"User-Agent": "curl"})
            local.write_bytes(urllib.request.urlopen(req, timeout=20).read())
        except Exception:
            fig, ax = plt.subplots(figsize=(3, 3)); ax.text(.5, .5, "image\nn/a", ha="center"); ax.axis("off")
            return _png(fig)
    return Response(local.read_bytes(), mimetype="image/jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/codec_rqvae_72x.pt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    model, ck = load(args.ckpt, device)
    avg, texts, imgs = ThingsEEG(split="test").trial_averaged()
    S.update(model=model, te=avg.to(device), concepts=texts, img_paths=imgs,
             meta=ck, ratio=ck["final_val"]["ratio"])
    print(f"serving RQ-VAE viewer ({args.ckpt}, {S['ratio']:.0f}x, MSE {ck['final_val']['mse']:.4f}) "
          f"on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
