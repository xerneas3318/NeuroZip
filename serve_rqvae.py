"""
serve_rqvae.py — port-8011 viewer for the RQ-VAE codec.

Self-contained Flask app: pick a held-out test concept, see the trial-averaged
ACTUAL EEG vs the RQ-VAE RECONSTRUCTION (heatmaps + a channel trace) and the
per-epoch MSE. No CLIP / open_clip needed — pure reconstruction viewing.

Run:
    python serve_rqvae.py --port 8011 --ckpt checkpoints/codec_rqvae_72x.pt
    # then open http://localhost:8011/
"""

import argparse
import base64
import io
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response

from data import ThingsEEG
from rqvae import EEGCodecRQ

app = Flask(__name__)
S = {}


def load(ckpt, device):
    ck = torch.load(ckpt, weights_only=False, map_location=device)
    c = ck["config"]
    m = EEGCodecRQ(c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"],
                   codebook_size=c["codebook_size"], num_quantizers=c["num_quantizers"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck


@app.route("/")
def index():
    opts = "".join(f'<option value="{i}">{i:03d} — {t}</option>'
                   for i, t in enumerate(S["concepts"]))
    fv = S["meta"]["final_val"]; cfg = S["meta"]["config"]
    return f"""<!doctype html><html><head><meta charset=utf-8><title>NeuroZip RQ-VAE</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#fafafa;color:#1a1a1a}}
h1{{margin-bottom:2px}}.sub{{color:#666;margin-top:0}}select{{padding:6px;font-size:15px}}
.card{{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:8px;margin-top:14px}}
img{{width:100%;border-radius:6px}}.badges{{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}}
.b{{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:8px 12px}}.b .n{{font-size:1.3em;font-weight:700}}.b .l{{color:#666;font-size:.85em}}</style></head><body>
<h1>NeuroZip — RQ-VAE reconstruction</h1>
<p class=sub>Residual vector-quantized latent. Held-out, trial-averaged test EEG.</p>
<div class=badges>
 <div class=b><div class=n>{S['ratio']:.0f}×</div><div class=l>compression vs float16</div></div>
 <div class=b><div class=n>{fv['mse']:.4f}</div><div class=l>test-avg MSE</div></div>
 <div class=b><div class=n>{fv.get('var_exp',0)*100:.0f}%</div><div class=l>variance explained</div></div>
 <div class=b><div class=n>D={cfg['num_quantizers']} K={cfg['codebook_size']}</div><div class=l>residual codebooks</div></div>
 <div class=b><div class=n>{cfg['c_lat']}×32</div><div class=l>latent (tokens×dim)</div></div>
</div>
<label>concept: <select id=sel onchange=upd()>{opts}</select></label>
<div class=card><img id=fig></div>
<script>
function upd(){{document.getElementById('fig').src='/api/recon?idx='+document.getElementById('sel').value+'&t='+Date.now()}}
upd();
</script></body></html>"""


@app.route("/api/recon")
def recon():
    i = int(request.args.get("idx", 0))
    x = S["te"][i:i+1]
    with torch.no_grad():
        xh, _ = S["model"].compress_then_reconstruct(x)
    xo = x[0].cpu().numpy(); xr = xh[0].cpu().numpy(); diff = xo - xr
    mse = float(((xo - xr) ** 2).mean())
    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    vmax = float(np.abs(np.concatenate([xo, xr])).max())
    ax[0, 0].imshow(xo, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[0, 0].set_title(f"ACTUAL — {S['concepts'][i]}")
    ax[0, 1].imshow(xr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax); ax[0, 1].set_title("RQ-VAE RECONSTRUCTION")
    dmax = float(np.abs(diff).max()) or 1.0
    ax[1, 0].imshow(diff, aspect="auto", cmap="seismic", vmin=-dmax, vmax=dmax); ax[1, 0].set_title(f"DIFFERENCE  (MSE {mse:.4f})")
    ch = 28
    ax[1, 1].plot(xo[ch], label="actual", color="0.4"); ax[1, 1].plot(xr[ch], label="recon", color="tab:blue")
    ax[1, 1].set_title(f"channel {ch}"); ax[1, 1].legend(fontsize=8)
    for a in ax.flat[:3]:
        a.set_xlabel("time"); a.set_ylabel("channel")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100); plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/codec_rqvae_72x.pt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    model, ck = load(args.ckpt, device)
    avg, texts, _ = ThingsEEG(split="test").trial_averaged()
    S.update(model=model, te=avg.to(device), concepts=texts, meta=ck, ratio=ck["final_val"]["ratio"])
    print(f"serving RQ-VAE ({args.ckpt}, {S['ratio']:.0f}x, MSE {ck['final_val']['mse']:.4f}) "
          f"on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
