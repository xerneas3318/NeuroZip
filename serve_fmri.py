"""
serve_fmri.py - NeuroZip fMRI viewer + semantic search (port 8012).

Self-contained: everything is baked into data_fmri/artifacts/sub-*.pt (the held-out
IT-cortex responses x2, both codec weights, the frozen fMRI->CLIP judge W, and the
CLIP image gallery). No dataset download needed.

For a chosen subject + concept it renders, like the criticism-2 viewer, four panels:
  INITIAL (raw IT pattern) | LATENT (compressed code) | DECOMPRESSED | DIFF (|initial-recon|)
for the selected codec, plus the SEMANTIC panel: decode each codec's reconstruction
through the frozen judge and rank the 100 concepts (fidelity codec vs the task-aware
"actual" codec). 2000s-gov-website aesthetic.

  FIDELITY codec = reconstruction (MSE) only        -> never sees CLIP
  ACTUAL  codec  = MSE + CLIP loss through the judge -> stays text-searchable

Run:  ./serve_fmri.sh   (or python serve_fmri.py)
"""
import argparse, base64, io, json, math, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fmri_neurozip import Codec

S = {}
ART = Path("data_fmri/artifacts")


def load_subject(subj):
    a = torch.load(ART / f"{subj}.pt", weights_only=False)
    V, lat = a["V"], a["latent"]
    fid = Codec(V, lat); fid.load_state_dict(a["fidelity"]); fid.eval()
    act = Codec(V, lat); act.load_state_dict(a["actual"]); act.eval()
    return {"concepts": a["concepts"], "metrics": a["metrics"], "V": V, "latent": lat,
            "x2": a["x2"].float(), "W": a["W"].float(),
            "gallery": F.normalize(a["gallery"].float(), dim=1),
            "codec": {"fidelity": fid, "actual": act}}


def _grid(vec):
    """Lay a 1-D voxel/latent vector out as a near-square 2-D grid for imshow."""
    v = np.asarray(vec, dtype=np.float32); n = v.size
    cols = int(math.ceil(math.sqrt(n))); rows = int(math.ceil(n / cols))
    g = np.full(rows * cols, np.nan, dtype=np.float32); g[:n] = v
    return g.reshape(rows, cols)


def _panel(ax, arr, title, vmax, cmap="RdBu_r", vmin=None):
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=(-vmax if vmin is None else vmin), vmax=vmax)
    ax.set_title(title, fontsize=10, loc="left", color="#003366", fontweight="bold", pad=4,
                 family="monospace")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#888")
    return im


def figure(subj, idx, codec_name):
    d = S["subj"][subj]; x = d["x2"][idx:idx + 1]
    c = d["codec"][codec_name]
    with torch.no_grad():
        z = torch.round(c.e(x)).squeeze(0).cpu().numpy()      # latent code
        recon = c(x).squeeze(0).cpu().numpy()                 # decompressed
    raw = x.squeeze(0).cpu().numpy()
    diff = np.abs(raw - recon)
    vmax = float(np.nanmax(np.abs([raw, recon])))             # shared scale init/recon
    fig, ax = plt.subplots(1, 4, figsize=(11.2, 3.1), gridspec_kw={"wspace": 0.18})
    _panel(ax[0], _grid(raw), "INITIAL  (raw IT pattern)", vmax)
    _panel(ax[1], _grid(z), f"LATENT  ({z.size} codes)", float(np.nanmax(np.abs(z)) or 1), cmap="PuOr")
    _panel(ax[2], _grid(recon), f"DECOMPRESSED  ({codec_name})", vmax)
    _panel(ax[3], _grid(diff), f"DIFF  |init-recon|  mse={float((diff**2).mean()):.3f}",
           float(np.nanmax(diff) or 1), cmap="magma", vmin=0)
    fig.patch.set_facecolor("#f4f0e6")
    for a in ax: a.set_facecolor("#f4f0e6")
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                                    facecolor="#f4f0e6"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def semantic(subj, idx, k=8):
    d = S["subj"][subj]; x = d["x2"][idx:idx + 1]; out = {}
    with torch.no_grad():
        for name, c in d["codec"].items():
            emb = F.normalize(c(x) @ d["W"], dim=1)            # recon -> CLIP space
            sims = (emb @ d["gallery"].t()).squeeze(0)
            top = sims.argsort(descending=True)[:k].tolist()
            out[name] = [{"concept": d["concepts"][i], "sim": float(sims[i])} for i in top]
    return out


CSS = """*{box-sizing:border-box}body{font-family:Arial,Verdana,sans-serif;background:#f4f0e6;color:#000;margin:0}
.bar{background:#003366;color:#fff;padding:10px 18px;font-weight:bold;text-transform:uppercase;letter-spacing:1px;border-bottom:3px solid #002244}
.bar span{color:#ffcc33}.wrap{max-width:1180px;margin:0 auto;padding:18px}.sub{font-size:.9em;margin:6px 0 14px}
.kpis{display:flex;border:2px solid #003366;margin:12px 0;background:#fff}
.kpi{flex:1;padding:9px 12px;border-right:1px solid #888}.kpi:last-child{border-right:0}
.kpi .v{font-family:"Courier New",monospace;font-size:1.5em;font-weight:bold}.kpi .v.us{color:#b23b34}
.kpi .k{font-size:.7em;text-transform:uppercase;color:#444;letter-spacing:.5px}
.controls{display:flex;gap:10px;align-items:center;margin:14px 0;flex-wrap:wrap}
input,select,button{font-family:Arial;font-size:15px;padding:7px 10px;border:2px solid #003366;background:#fff;border-radius:0}
button{background:#003366;color:#fff;cursor:pointer;font-weight:bold;text-transform:uppercase}
.seg button{border-left:0}.seg button.on{background:#b23b34}
.fig{border:2px solid #003366;background:#fff;padding:6px;margin:4px 0}.fig img{width:100%;display:block}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}.card{border:2px solid #003366;background:#fff}
.card h3{margin:0;background:#003366;color:#fff;padding:7px 12px;font-size:.85em;text-transform:uppercase;letter-spacing:1px}
.card.us h3{background:#b23b34}table{width:100%;border-collapse:collapse;font-size:.92em}
td{padding:5px 12px;border-bottom:1px solid #ddd}td.r{font-family:"Courier New",monospace;text-align:right;color:#555}
tr.hit td{background:#fff3cf;font-weight:bold}tr.hit td:first-child::after{content:" \\2713";color:#1a7a1a}
.rank{font-family:"Courier New",monospace;color:#888;width:34px}
.note{font-size:.82em;color:#444;margin-top:16px;border-top:1px solid #888;padding-top:10px}"""


def page():
    d0 = S["subj"][S["default"]]
    opts = "".join(f"<option value={i}>{c}</option>" for i, c in enumerate(d0["concepts"]))
    subs = "".join(f"<option>{s}</option>" for s in S["subjects"])
    return f"""<!doctype html><html><head><meta charset=utf-8><title>NeuroZip fMRI viewer</title>
<style>{CSS}</style></head><body>
<div class=bar>Neuro<span>Zip</span> &nbsp;//&nbsp; fMRI viewer + semantic search &nbsp;//&nbsp; IT cortex, THINGS-data</div>
<div class=wrap>
<div class=sub>Pick a concept. We compress its held-out IT-cortex pattern to {d0['latent']} codes
({d0['metrics']['ratio']}x), then show the <b>initial / latent / decompressed / diff</b> panels and
how each codec's reconstruction decodes through the frozen fMRI&rarr;CLIP judge.</div>
<div class=controls>subject <select id=subj onchange=load()>{subs}</select>
concept <select id=ci onchange=go()>{opts}</select>
<span class=seg><button id=bact class=on onclick="setc('actual')">actual</button><button id=bfid onclick="setc('fidelity')">fidelity</button></span></div>
<div class=kpis id=kpis></div>
<div class=fig><img id=fig></div>
<div class=cols><div class=card><h3>Fidelity codec &mdash; MSE only (CLIP-blind)</h3><table id=fid></table></div>
<div class=card us><h3>Actual codec &mdash; backprops through CLIP</h3><table id=act></table></div></div>
<p class=note>Held-out eval (reps 7&ndash;12 averaged), 100-way retrieval, chance top1=1%. INITIAL is the raw
normalized IT pattern ({d0['V']} voxels); DIFF is |initial &minus; reconstruction|. The <b>actual</b> codec was
trained with a CLIP loss through the frozen judge; <b>fidelity</b> only minimized MSE. Same latent / same bitrate.
&check; = the concept actually shown to the subject.</p></div>
<script>
let codec='actual';
function setc(c){{codec=c;bact.className=c=='actual'?'on':'';bfid.className=c=='fidelity'?'on':'';go();}}
async function load(){{const m=(await (await fetch('/api/meta?subj='+subj.value)).json()).metrics;
kpis.innerHTML=`<div class=kpi><div class=v>${{m.ratio}}x</div><div class=k>compression</div></div>`+
`<div class=kpi><div class=v>${{(m.judge.top5*100).toFixed(0)}}%</div><div class=k>uncompressed top5</div></div>`+
`<div class=kpi><div class=v>${{(m.fidelity.top5*100).toFixed(0)}}%</div><div class=k>fidelity top5</div></div>`+
`<div class=kpi><div class="v us">${{(m.actual.top5*100).toFixed(0)}}%</div><div class=k>actual top5</div></div>`+
`<div class=kpi><div class="v us">+${{((m.actual.top5-m.fidelity.top5)*100).toFixed(0)}}pp</div><div class=k>actual lift</div></div>`;go();}}
async function go(){{const d=await (await fetch('/api/view?subj='+subj.value+'&ci='+ci.value+'&codec='+codec)).json();
fig.src=d.fig;const gold=d.gold;
const row=(x,i)=>`<tr class="${{x.concept==gold?'hit':''}}"><td class=rank>${{i+1}}</td><td>${{x.concept}}</td><td class=r>${{x.sim.toFixed(3)}}</td></tr>`;
fid.innerHTML=d.semantic.fidelity.map(row).join('');act.innerHTML=d.semantic.actual.map(row).join('');}}
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); qs = urllib.parse.parse_qs(u.query)
        g = lambda k, dv="": qs.get(k, [dv])[0]
        if u.path == "/":
            self._send(page(), "text/html; charset=utf-8")
        elif u.path == "/api/meta":
            self._send(json.dumps({"metrics": S["subj"][g("subj", S["default"])]["metrics"]}))
        elif u.path == "/api/view":
            subj = g("subj", S["default"]); idx = int(g("ci", "0")); codec = g("codec", "actual")
            self._send(json.dumps({"fig": figure(subj, idx, codec),
                                   "semantic": semantic(subj, idx),
                                   "gold": S["subj"][subj]["concepts"][idx]}))
        else:
            self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8012)
    args = ap.parse_args()
    subs = sorted(p.stem for p in ART.glob("sub-*.pt"))
    S["subjects"] = subs; S["default"] = subs[0]
    S["subj"] = {s: load_subject(s) for s in subs}
    print(f"serving NeuroZip-fMRI viewer on http://{args.host}:{args.port}/  subjects={subs}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
