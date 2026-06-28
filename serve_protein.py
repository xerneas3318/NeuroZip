"""
serve_protein.py — port-8011 viewer for the protein v4 codec.

For a held-out protein: the ORIGINAL one-hot ("before latent construction"), the
ENCODED latent codes, the RECONSTRUCTION, and the DIFFERENCE (original - recon
overlay subtraction) — plus all stats (per-residue accuracy, MSE, bits/residue,
compression ratio, mismatches) and the original-vs-reconstructed sequence with
mismatches highlighted. A compression-tier slider swaps codecs.

Run:  ./serve_protein.sh   # or python serve_protein.py --port 8011
"""
import argparse, io, os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response

from protein_data import load_proteins, to_onehot, PAD, AA_ORDER
from protein_codec import ProteinCodec

app = Flask(__name__)
S = {}
SRC_BITS = 20 * 250 * 16

TIERS = [   # ordered by compression ratio (fidelity -> compression)
    ("high fidelity", "checkpoints/protein_lr0.02.pt"),
    ("medium",        "checkpoints/protein_lr0.1.pt"),
    ("high compression", "checkpoints/protein_lr0.5.pt"),
]

CSS = """
:root{--bg:#f6f7fa;--surface:#fff;--border:#e3e6ee;--ink:#0f1322;--ink-2:#4a5163;
--ink-muted:#8a92a8;--accent:#1f9d55;--bad:#d62728;--shadow:0 1px 2px rgba(15,19,34,.04),0 8px 24px rgba(15,19,34,.05);--radius:14px}
*{box-sizing:border-box}body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:0 16px 60px}
.wrap{max-width:1120px;margin:0 auto}header{display:flex;align-items:baseline;gap:14px;padding:24px 0 4px}
h1{margin:0;font-size:1.7em;letter-spacing:-.5px}h1 span{color:var(--accent)}.sub{color:var(--ink-2);margin:2px 0 0}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}
.cell{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:11px 16px;box-shadow:var(--shadow);min-width:120px}
.cell .v{font-size:1.35em;font-weight:700}.cell .v.us{color:var(--accent)}.cell .k{color:var(--ink-muted);font-size:.74em;text-transform:uppercase;letter-spacing:.5px}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:6px 0 14px}
.tierbox{flex:1;min-width:260px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:10px 16px;box-shadow:var(--shadow)}
.tierbox label{color:var(--ink-muted);font-size:.78em;text-transform:uppercase;letter-spacing:.5px;display:flex;justify-content:space-between}
.tierbox label b{color:var(--accent);text-transform:none;letter-spacing:0}
input[type=range]{width:100%}select,button{font-size:15px;padding:8px 12px;border-radius:10px;border:1px solid var(--border);background:var(--surface)}
button{cursor:pointer;font-weight:600}.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:12px;box-shadow:var(--shadow);margin-top:16px}
.card h3{margin:2px 0 8px;font-size:.92em;color:var(--ink-2)}.card img{width:100%;border-radius:10px;display:block}
.seq{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;line-height:1.5;word-break:break-all;background:#fafbfc;border:1px solid var(--border);border-radius:8px;padding:10px}
.seq .mm{background:var(--bad);color:#fff;border-radius:2px}.seqlab{color:var(--ink-muted);font-size:.78em;text-transform:uppercase;margin:6px 0 2px}
.note{color:var(--ink-muted);font-size:.85em;margin-top:16px}
"""


def load_codec(ckpt, device):
    ck = torch.load(ckpt, weights_only=False, map_location=device)
    c = ck["config"]
    m = ProteinCodec(n_aa=c["n_aa"], c_lat=c["c_lat"], hidden=c["hidden"], n_attn=c["n_attn"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    fv = ck["final_val"]
    bits = SRC_BITS / fv["ratio"]
    return {"model": m, "fv": fv, "ratio": fv["ratio"], "bytes": bits / 8}


def tier(ti):
    return S["tiers"][max(0, min(ti, len(S["tiers"]) - 1))]


def _compute(ti, i):
    e = tier(ti); m = e["model"]; idx = torch.from_numpy(S["va_idx"][i:i+1]).to(S["device"])
    L = int(S["va_lens"][i])
    x = to_onehot(idx, S["device"])
    with torch.no_grad():
        y = torch.round(m.encoder(x))                 # latent codes
        logits = m.decoder(y)
        probs = F.softmax(logits, 1)
    gold = idx[0].cpu().numpy(); pred = logits.argmax(1)[0].cpu().numpy()
    mask = gold != PAD
    acc = float((pred[mask] == gold[mask]).mean())
    mse = float(((x[0] - probs[0]) ** 2)[:, :L].mean().cpu())
    return {"L": L, "x": x[0].cpu().numpy(), "lat": y[0].cpu().numpy(),
            "probs": probs[0].cpu().numpy(), "pred": pred, "gold": gold,
            "acc": acc, "mse": mse, "mism": int((pred[mask] != gold[mask]).sum())}


@app.route("/api/tiers")
def api_tiers():
    return jsonify([{"label": l, "ratio": e["ratio"], "acc": e["fv"]["acc"],
                     "bits_per_residue": e["fv"]["bits_per_residue"], "bytes": e["bytes"]}
                    for l, e in zip(S["tier_labels"], S["tiers_e"])])


@app.route("/api/info")
def api_info():
    i = int(request.args.get("idx", 0)); ti = int(request.args.get("tier", len(S["tiers"]) - 1))
    d = _compute(ti, i)
    seq_o = "".join(AA_ORDER[a] for a in d["gold"][:d["L"]])
    seq_r = "".join(AA_ORDER[a] if a < 20 else "-" for a in d["pred"][:d["L"]])
    mm = [p for p in range(d["L"]) if d["gold"][p] != d["pred"][p]]
    e = tier(ti)
    return jsonify({"L": d["L"], "acc": d["acc"], "mse": d["mse"], "mismatches": d["mism"],
                    "bits_per_residue": e["fv"]["bits_per_residue"], "ratio": e["ratio"],
                    "bytes": e["bytes"], "seq_o": seq_o, "seq_r": seq_r, "mm": mm})


def _png(fig):
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=108); plt.close(fig)
    return Response(buf.getvalue(), mimetype="image/png")


@app.route("/api/panels")
def panels():
    i = int(request.args.get("idx", 0)); ti = int(request.args.get("tier", len(S["tiers"]) - 1))
    d = _compute(ti, i); L = d["L"]
    xo = d["x"][:, :L]; xr = d["probs"][:, :L]
    rec_oh = np.zeros_like(xo)
    for p in range(L):
        if d["pred"][p] < 20: rec_oh[d["pred"][p], p] = 1
    diff = xo - rec_oh
    fig, ax = plt.subplots(4, 1, figsize=(11, 8.2), gridspec_kw={"hspace": 0.55})
    ax[0].imshow(xo, aspect="auto", cmap="Greens", interpolation="nearest")
    ax[0].set_title("BEFORE latent — original protein one-hot (20 amino acids × residues)", fontsize=10, loc="left")
    ax[1].imshow(d["lat"], aspect="auto", cmap="PuOr", interpolation="nearest")
    ax[1].set_title(f"ENCODED latent — {d['lat'].shape[0]}×{d['lat'].shape[1]} integer codes (the compressed form)", fontsize=10, loc="left")
    ax[1].set_xlabel("latent position")
    ax[2].imshow(xr, aspect="auto", cmap="Greens", interpolation="nearest")
    ax[2].set_title(f"RECONSTRUCTION — decoded probabilities  ({d['acc']*100:.1f}% residues correct)", fontsize=10, loc="left")
    dm = 1.0
    ax[3].imshow(diff, aspect="auto", cmap="seismic", vmin=-dm, vmax=dm, interpolation="nearest")
    ax[3].set_title(f"DIFFERENCE — original − reconstruction  ({d['mism']} mismatched residues)", fontsize=10, loc="left")
    for a in (ax[0], ax[2], ax[3]):
        a.set_yticks(range(20)); a.set_yticklabels(list(AA_ORDER), fontsize=5); a.set_xlabel("residue")
    return _png(fig)


@app.route("/")
def index():
    opts = "".join(f'<option value="{i}">{i:04d} — len {int(S["va_lens"][i])}</option>'
                   for i in range(min(len(S["va_idx"]), 1500)))
    n = len(S["tiers"])
    return f"""<!doctype html><html><head><meta charset=utf-8><title>NeuroZip · protein codec</title>
<style>{CSS}</style></head><body><div class=wrap>
<header><h1>Neuro<span>Zip</span></h1><div class=sub>v4 codec on PROTEINS — same architecture as EEG (63→20 channels) · held-out UniProt sequences</div></header>
<div class=stats id=stats></div>
<div class=controls>
 <label>protein&nbsp;<select id=sel onchange=upd()>{opts}</select></label>
 <button onclick=rnd()>Random</button>
 <div class=tierbox><label>compression tier <b id=tierlab>—</b></label>
  <input id=tier type=range min=0 max={n-1} value={n-1} step=1 oninput=upd()></div>
</div>
<div class=card><h3>original · latent · reconstruction · difference</h3><img id=panels></div>
<div class=card><h3>sequence — original vs reconstruction (mismatches in red)</h3>
 <div class=seqlab>original</div><div class=seq id=seqo></div>
 <div class=seqlab>reconstruction</div><div class=seq id=seqr></div></div>
<p class=note>The protein is a (20 × length) one-hot "image", exactly analogous to the (63 × 250) EEG epoch.
Latent, scalar quantizer and Laplace rate model are byte-for-byte the same as the EEG v4 codec.</p>
<script>
let TIERS=[];
async function loadTiers(){{TIERS=await (await fetch('/api/tiers')).json();}}
function mark(seq,mm){{const s=new Set(mm);return [...seq].map((c,p)=>s.has(p)?`<span class=mm>${{c}}</span>`:c).join('');}}
async function info(i,ti){{const d=await (await fetch('/api/info?idx='+i+'&tier='+ti)).json();
 const t=TIERS[ti]||{{}};
 document.getElementById('stats').innerHTML=
  `<div class=cell><div class="v us">${{(d.acc*100).toFixed(1)}}%</div><div class=k>per-residue accuracy</div></div>`+
  `<div class=cell><div class=v>${{d.mse.toFixed(4)}}</div><div class=k>recon MSE (one-hot)</div></div>`+
  `<div class=cell><div class=v>${{d.mismatches}}</div><div class=k>mismatched residues</div></div>`+
  `<div class=cell><div class="v us">${{d.ratio.toFixed(0)}}×</div><div class=k>compression vs fp16</div></div>`+
  `<div class=cell><div class=v>${{d.bits_per_residue.toFixed(2)}}</div><div class=k>bits / residue</div></div>`+
  `<div class=cell><div class=v>${{d.bytes.toFixed(0)}} B</div><div class=k>bytes / protein</div></div>`+
  `<div class=cell><div class=v>${{d.L}}</div><div class=k>length (residues)</div></div>`;
 document.getElementById('seqo').innerHTML=mark(d.seq_o,d.mm);
 document.getElementById('seqr').innerHTML=mark(d.seq_r,d.mm);
 document.getElementById('tierlab').textContent=(t.label||'')+' · '+(t.ratio?t.ratio.toFixed(0)+'×':'');}}
function upd(){{const i=document.getElementById('sel').value,ti=document.getElementById('tier').value,t=Date.now();
 document.getElementById('panels').src='/api/panels?idx='+i+'&tier='+ti+'&t='+t; info(i,ti);}}
function rnd(){{const s=document.getElementById('sel');s.value=Math.floor(Math.random()*s.options.length);upd();}}
loadTiers().then(upd);
</script></div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    tiers, tiers_e = [], []
    for label, ckpt in TIERS:
        if os.path.exists(ckpt):
            e = load_codec(ckpt, device); tiers.append((label, e)); tiers_e.append(e)
    assert tiers, "no protein checkpoints"
    tiers.sort(key=lambda t: t[1]["ratio"])
    tiers_e = [t[1] for t in tiers]
    _, (va_idx, va_lens), _ = load_proteins()
    S.update(device=device, tiers=tiers_e, tiers_e=tiers_e,
             tier_labels=[t[0] for t in tiers], va_idx=va_idx, va_lens=va_lens)
    print(f"serving protein viewer on http://{args.host}:{args.port}/  tiers={[t[0] for t in tiers]}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
