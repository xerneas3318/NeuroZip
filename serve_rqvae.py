"""
serve_rqvae.py — port-8011 viewer: RQ-VAE vs rian's v4, across compression tiers.

A compression-tier slider swaps the codecs (different ratios). For each concept:
the stimulus image, the trial-averaged ACTUAL EEG, and each codec's
reconstruction with PER-IMAGE MSE, plus per-model stats (MSE, bpp, bytes, ratio).
UI styled after rian's demo_clean.html. Self-contained (no open_clip / judge).

Run:  ./serve_rqvae.sh    # or python serve_rqvae.py --port 8011
"""
import argparse, io, os, urllib.request
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response

from data import ThingsEEG, N_CHANNELS, N_TIMES
from codec import EEGCodec
from rqvae import EEGCodecRQ

app = Flask(__name__)
S = {}
HF_IMG = "https://huggingface.co/datasets/Haitao999/things-eeg/resolve/main/Image_set/"
STIM_CACHE = Path("results/stimuli"); STIM_CACHE.mkdir(parents=True, exist_ok=True)
SRC_BITS = N_CHANNELS * N_TIMES * 16

# tier label -> {model name -> checkpoint path}.  Loaded if the file exists.
TIERS = [
    ("high compression", {"RQ-VAE": "checkpoints/codec_rqvae_200x.pt", "v4 (rian)": "checkpoints/codec_v4_200x.pt"}),
    ("medium",           {"RQ-VAE": "checkpoints/codec_rqvae_100x.pt", "v4 (rian)": "checkpoints/codec_v4_100x.pt"}),
    ("high fidelity",    {"RQ-VAE": "checkpoints/codec_rqvae_72x.pt",  "v4 (rian)": "checkpoints/codec_v4_fidelity.pt"}),
]

CSS = """
:root{--bg:#f6f7fa;--surface:#fff;--surface-2:#f0f2f7;--border:#e3e6ee;--ink:#0f1322;
--ink-2:#4a5163;--ink-muted:#8a92a8;--accent:#d62728;--neutral:#6b7388;--ok:#1f9d55;
--shadow:0 1px 2px rgba(15,19,34,.04),0 8px 24px rgba(15,19,34,.05);--radius:14px}
*{box-sizing:border-box}body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,sans-serif;
background:var(--bg);color:var(--ink);margin:0;padding:0 16px 60px}.wrap{max-width:1120px;margin:0 auto}
header{display:flex;align-items:baseline;gap:14px;padding:24px 0 4px}
h1{margin:0;font-size:1.7em;letter-spacing:-.5px}h1 span{color:var(--accent)}.sub{color:var(--ink-2);margin:2px 0 0}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
.cell{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:11px 16px;box-shadow:var(--shadow);min-width:120px}
.cell .v{font-size:1.3em;font-weight:700}.cell .v.us{color:var(--accent)}.cell .v.them{color:var(--neutral)}
.cell .k{color:var(--ink-muted);font-size:.74em;text-transform:uppercase;letter-spacing:.5px}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:6px 0 14px}
.tierbox{flex:1;min-width:260px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:10px 16px;box-shadow:var(--shadow)}
.tierbox label{color:var(--ink-muted);font-size:.78em;text-transform:uppercase;letter-spacing:.5px;display:flex;justify-content:space-between}
.tierbox label b{color:var(--accent);font-size:1.05em;text-transform:none;letter-spacing:0}
input[type=range]{width:100%}select,button{font-size:15px;padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:var(--surface)}
button{cursor:pointer;font-weight:600}button:hover{background:var(--surface-2)}
.row{display:grid;grid-template-columns:230px 1fr;gap:18px;align-items:start}
.stim{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px;box-shadow:var(--shadow)}
.stim img{width:100%;border-radius:10px;display:block}.stim .cap{color:var(--ink-muted);font-size:.82em;margin:6px 0 8px;text-align:center}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px;box-shadow:var(--shadow);margin-top:16px}
.card h3{margin:2px 0 8px;font-size:.92em;color:var(--ink-2)}.card img{width:100%;border-radius:10px;display:block}
table{width:100%;border-collapse:collapse;font-size:.9em}th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--border)}
th:first-child,td:first-child{text-align:left}thead th{color:var(--ink-muted);font-weight:600;font-size:.78em;text-transform:uppercase}
td.win{color:var(--ok);font-weight:700}.model-us{color:var(--accent);font-weight:700}.model-them{color:var(--neutral);font-weight:700}
.note{color:var(--ink-muted);font-size:.85em;margin-top:16px}
"""


def load_any(ckpt, device):
    ck = torch.load(ckpt, weights_only=False, map_location=device)
    c = ck["config"]
    if "num_quantizers" in c:
        m = EEGCodecRQ(c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"],
                       codebook_size=c["codebook_size"], num_quantizers=c["num_quantizers"]); kind = "rqvae"
    else:
        m = EEGCodec(c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"]); kind = "scalar"
    m.to(device); m.load_state_dict(ck["model"]); m.eval()
    fv = ck["final_val"]; bits = SRC_BITS / fv["ratio"]
    return {"model": m, "kind": kind, "ratio": fv["ratio"], "bits": bits,
            "bytes": bits / 8, "mse_all": fv["mse"]}


def recon(entry, x):
    with torch.no_grad():
        xh, _ = entry["model"].compress_then_reconstruct(x)
    return xh


def tier(i):
    return S["tiers"][max(0, min(i, len(S["tiers"]) - 1))]


@app.route("/api/tiers")
def api_tiers():
    out = []
    for label, models in S["tiers"]:
        rq = models.get("RQ-VAE")
        out.append({"label": label, "ratio": rq["ratio"] if rq else 0,
                    "models": {n: {"mse_all": e["mse_all"], "ratio": e["ratio"],
                                   "bytes": e["bytes"], "kind": e["kind"]} for n, e in models.items()}})
    return jsonify(out)


@app.route("/")
def index():
    opts = "".join(f'<option value="{i}">{i:03d} — {t}</option>' for i, t in enumerate(S["concepts"]))
    n = len(S["tiers"])
    return f"""<!doctype html><html><head><meta charset=utf-8><title>NeuroZip · RQ-VAE vs v4</title>
<style>{CSS}</style></head><body><div class=wrap>
<header><h1>Neuro<span>Zip</span></h1><div class=sub>RQ-VAE vs rian's v4 · across compression tiers · held-out, trial-averaged test EEG</div></header>
<div class=stats id=stats></div>
<div class=controls>
 <label>concept&nbsp;<select id=sel onchange=upd()>{opts}</select></label>
 <button onclick=rnd()>Random</button>
 <div class=tierbox><label>compression tier <b id=tierlab>—</b></label>
   <input id=tier type=range min=0 max={n-1} value={n-1} step=1 oninput=upd()></div>
</div>
<div class=row>
 <div class=stim><img id=stim><div class=cap id=stimcap>stimulus</div>
   <table id=info><thead><tr><th>model</th><th>MSE</th><th>bpp</th></tr></thead><tbody></tbody></table></div>
 <div class=card><h3 id=heat-h>actual vs reconstructions (normalized σ)</h3><img id=heat></div>
</div>
<div class=card><h3>RQ-VAE discrete latent codes · per-channel waveforms (actual vs models)</h3><img id=detail></div>
<p class=note>Slide the compression tier to swap codecs. Per-image MSE is computed live.
v4 (rian) = conv-only codec trained on single trials; it over-shoots amplitude on
trial-averaged EEG (higher MSE). bpp = bits per EEG sample; bytes = compressed epoch size.</p>
<script>
let TIERS=[];
async function loadTiers(){{TIERS=await (await fetch('/api/tiers')).json();}}
function fmt(x){{return x.toFixed(4);}}
function stats(){{const t=TIERS[+tier.value]||{{models:{{}}}};
 let cells=`<div class=cell><div class=v>${{t.ratio?t.ratio.toFixed(0):'—'}}×</div><div class=k>compression</div></div>`;
 for(const [n,m] of Object.entries(t.models)){{const c=m.kind=='rqvae'?'us':'them';
  cells+=`<div class=cell><div class="v ${{c}}">${{fmt(m.mse_all)}}</div><div class=k>${{n}} · MSE</div></div>`;}}
 document.getElementById('stats').innerHTML=cells;
 document.getElementById('tierlab').textContent=t.label+' · '+(t.ratio?t.ratio.toFixed(0)+'×':'');}}
async function info(i,ti){{const d=await (await fetch('/api/info?idx='+i+'&tier='+ti)).json();
 let best=Math.min(...Object.values(d.models).map(m=>m.mse));
 document.querySelector('#info tbody').innerHTML=Object.entries(d.models).map(([n,m])=>
  `<tr><td class="model-${{m.kind=='rqvae'?'us':'them'}}">${{n}}</td><td class="${{m.mse==best?'win':''}}">${{fmt(m.mse)}}</td><td>${{m.bpp.toFixed(3)}}</td></tr>`).join('');
 document.getElementById('heat-h').textContent='actual · '+Object.keys(d.models).join(' · ')+' (per-image MSE on each)';}}
function upd(){{const i=document.getElementById('sel').value,ti=document.getElementById('tier').value,t=Date.now();
 document.getElementById('heat').src=`/api/heat?idx=${{i}}&tier=${{ti}}&t=${{t}}`;
 document.getElementById('detail').src=`/api/detail?idx=${{i}}&tier=${{ti}}&t=${{t}}`;
 document.getElementById('stim').src='/api/stimulus?idx='+i+'&t='+t;
 document.getElementById('stimcap').textContent=document.getElementById('sel').selectedOptions[0].textContent.split('— ')[1];
 stats();info(i,ti);}}
function rnd(){{const s=document.getElementById('sel');s.value=Math.floor(Math.random()*s.options.length);upd();}}
loadTiers().then(upd);
</script></div></body></html>"""


@app.route("/api/info")
def api_info():
    i = int(request.args.get("idx", 0)); ti = int(request.args.get("tier", len(S["tiers"]) - 1))
    x = S["te"][i:i+1]; out = {"concept": S["concepts"][i], "models": {}}
    for name, e in tier(ti)[1].items():
        xh = recon(e, x); mse = float(((xh - x) ** 2).mean())
        out["models"][name] = {"mse": mse, "bpp": e["bits"] / (N_CHANNELS * N_TIMES),
                               "ratio": e["ratio"], "bytes": e["bytes"], "kind": e["kind"]}
    return jsonify(out)


def _png(fig):
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=108); plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")


@app.route("/api/heat")
def heat():
    i = int(request.args.get("idx", 0)); ti = int(request.args.get("tier", len(S["tiers"]) - 1))
    x = S["te"][i:i+1]; xo = x[0].cpu().numpy()
    panels = [("ACTUAL — " + S["concepts"][i], xo)]
    for name, e in tier(ti)[1].items():
        xr = recon(e, x)[0].cpu().numpy()
        panels.append((f"{name}  MSE={float(((xo-xr)**2).mean()):.4f}", xr))
    vmax = float(np.max([np.abs(p[1]).max() for p in panels])); nn = len(panels)
    fig, ax = plt.subplots(1, nn, figsize=(4.0 * nn, 3.4)); ext = [-200, 996, 63, 0]
    for a, (t, arr) in zip(np.atleast_1d(ax), panels):
        a.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=ext)
        a.set_title(t, fontsize=10); a.set_xlabel("time after stim (ms)"); a.set_ylabel("channel")
    return _png(fig)


@app.route("/api/detail")
def detail():
    i = int(request.args.get("idx", 0)); ti = int(request.args.get("tier", len(S["tiers"]) - 1))
    x = S["te"][i:i+1]; xo = x[0].cpu().numpy(); models = tier(ti)[1]
    rq = models.get("RQ-VAE")
    fig = plt.figure(figsize=(12, 3.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 1.6], wspace=0.32)
    axc = fig.add_subplot(gs[0])
    if rq is not None:
        with torch.no_grad():
            z = rq["model"].encoder(x).transpose(1, 2); _, codes, _ = rq["model"].rvq(z)
        axc.imshow(codes[0].cpu().numpy().T, aspect="auto", cmap="viridis", interpolation="nearest")
        axc.set_title(f"RQ codes ({codes.shape[-1]}×32)"); axc.set_xlabel("token"); axc.set_ylabel("quantizer")
    else:
        axc.axis("off")
    recons = {name: recon(e, x)[0].cpu().numpy() for name, e in models.items()}
    t = np.linspace(-200, 996, N_TIMES); col = {"rqvae": "#d62728", "scalar": "#6b7388"}
    for k, ch in enumerate([18, 47]):
        ax = fig.add_subplot(gs[k + 1]); ax.plot(t, xo[ch], color="0.3", lw=1.3, label="actual")
        for name, e in models.items():
            ax.plot(t, recons[name][ch], color=col[e["kind"]], lw=1.0,
                    ls="-" if e["kind"] == "rqvae" else "--", label=name)
        ax.axvline(0, color="0.8", lw=.7); ax.set_title(f"channel {ch}"); ax.set_xlabel("time after stim (ms)")
        if k == 0: ax.legend(fontsize=8)
    return _png(fig)


@app.route("/api/stimulus")
def stimulus():
    i = int(request.args.get("idx", 0)); path = S["img_paths"][i]
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
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    tiers = []
    for label, models in TIERS:
        loaded = {}
        for name, path in models.items():
            if os.path.exists(path):
                loaded[name] = load_any(path, device)
        if loaded:
            tiers.append((label, loaded))
    assert tiers, "no tier checkpoints found"
    avg, texts, imgs = ThingsEEG(split="test").trial_averaged()
    S.update(tiers=tiers, te=avg.to(device), concepts=texts, img_paths=imgs)
    print(f"serving on http://{args.host}:{args.port}/  tiers=" +
          ", ".join(f"{l}({list(m)})" for l, m in tiers))
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
