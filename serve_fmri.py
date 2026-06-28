"""
serve_fmri.py — NeuroZip fMRI semantic-search demo (port 8012). Stdlib only.

Type a word -> its CLIP text embedding -> rank the 100 held-out IT-cortex fMRI
responses by how close each codec's RECONSTRUCTION decodes to that word.

  FIDELITY codec  = reconstruction (MSE) only            -> never sees CLIP
  ACTUAL  codec   = MSE + CLIP loss backpropagated through the frozen judge

Same compression for both. The actual codec keeps the brain response text-
searchable; the fidelity one throws that away. 2000s-gov-website aesthetic.

Run:  ./serve_fmri.sh   (or python serve_fmri.py)
"""
import argparse, json, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch, torch.nn.functional as F

from fmri_neurozip import Codec

S = {}
ART = Path("data_fmri/artifacts")
FEAT = "data/Preprocessed_data_250Hz_whiten/ViT-B-32_features_{}.pt"


def load_subject(subj):
    a = torch.load(ART / f"{subj}.pt", weights_only=False)
    V, lat = a["V"], a["latent"]
    fid = Codec(V, lat); fid.load_state_dict(a["fidelity"]); fid.eval()
    act = Codec(V, lat); act.load_state_dict(a["actual"]); act.eval()
    W, x2 = a["W"], a["x2"]
    judge = lambda x: x @ W
    with torch.no_grad():
        emb = {"fidelity": F.normalize(judge(fid(x2)), dim=1),
               "actual": F.normalize(judge(act(x2)), dim=1),
               "uncompressed": F.normalize(judge(x2), dim=1)}
    return {"concepts": a["concepts"], "metrics": a["metrics"], "emb": emb}


def query_feature(q):
    q = q.strip().lower(); feats = S["text_feats"]
    if q in feats: return feats[q], q
    hits = sorted([c for c in feats if q and (q in c or c in q)], key=len)
    return (feats[hits[0]], hits[0]) if hits else (None, None)


def rank(subj, qf, codec, k=8):
    e = S["subj"][subj]["emb"][codec]; sims = (e @ qf).squeeze(-1)
    idx = sims.argsort(descending=True)[:k].tolist()
    cs = S["subj"][subj]["concepts"]
    return [{"concept": cs[i], "sim": float(sims[i])} for i in idx]


CSS = """*{box-sizing:border-box}body{font-family:Arial,Verdana,sans-serif;background:#f4f0e6;color:#000;margin:0}
.bar{background:#003366;color:#fff;padding:10px 18px;font-weight:bold;text-transform:uppercase;letter-spacing:1px;border-bottom:3px solid #002244}
.bar span{color:#ffcc33}.wrap{max-width:1080px;margin:0 auto;padding:18px}.sub{font-size:.9em;margin:6px 0 16px}
.kpis{display:flex;border:2px solid #003366;margin:14px 0;background:#fff}
.kpi{flex:1;padding:10px 12px;border-right:1px solid #888}.kpi:last-child{border-right:0}
.kpi .v{font-family:"Courier New",monospace;font-size:1.5em;font-weight:bold}.kpi .v.us{color:#b23b34}
.kpi .k{font-size:.7em;text-transform:uppercase;color:#444;letter-spacing:.5px}
.controls{display:flex;gap:10px;align-items:center;margin:14px 0}
input,select,button{font-family:Arial;font-size:15px;padding:7px 10px;border:2px solid #003366;background:#fff;border-radius:0}
button{background:#003366;color:#fff;cursor:pointer;font-weight:bold;text-transform:uppercase}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{border:2px solid #003366;background:#fff}
.card h3{margin:0;background:#003366;color:#fff;padding:7px 12px;font-size:.85em;text-transform:uppercase;letter-spacing:1px}
.card.us h3{background:#b23b34}table{width:100%;border-collapse:collapse;font-size:.92em}
td{padding:5px 12px;border-bottom:1px solid #ddd}td.r{font-family:"Courier New",monospace;text-align:right;color:#555}
tr.hit td{background:#fff3cf;font-weight:bold}tr.hit td:first-child::after{content:" \\2713";color:#1a7a1a}
.rank{font-family:"Courier New",monospace;color:#888;width:34px}
.note{font-size:.82em;color:#444;margin-top:16px;border-top:1px solid #888;padding-top:10px}"""


def page():
    cs = sorted(S["subj"][S["default"]]["concepts"])
    opts = "".join(f"<option>{c}</option>" for c in cs)
    subs = "".join(f"<option>{s}</option>" for s in S["subjects"])
    return f"""<!doctype html><html><head><meta charset=utf-8><title>NeuroZip fMRI · semantic search</title>
<style>{CSS}</style></head><body>
<div class=bar>Neuro<span>Zip</span> &nbsp;//&nbsp; fMRI semantic search &nbsp;//&nbsp; IT cortex, THINGS-data</div>
<div class=wrap>
<div class=sub>Type a concept. We rank the 100 held-out brain responses by how each codec's
<b>reconstruction</b> decodes through the frozen fMRI&rarr;CLIP judge. Both codecs use the same compression.</div>
<div class=controls>subject <select id=subj onchange=load()>{subs}</select>
<input id=q list=cl placeholder="e.g. accordion" value="accordion" onkeydown="if(event.key=='Enter')go()">
<datalist id=cl>{opts}</datalist><button onclick=go()>Search</button></div>
<div class=kpis id=kpis></div>
<div class=cols><div class=card><h3>Fidelity codec &mdash; MSE only (CLIP-blind)</h3><table id=fid></table></div>
<div class=card us><h3>Actual codec &mdash; backprops through CLIP</h3><table id=act></table></div></div>
<p class=note>Held-out eval (reps 7&ndash;12), 100-way retrieval, chance top1=1%. The <b>actual</b> codec
trained with a CLIP loss through a frozen fMRI&rarr;CLIP judge; the <b>fidelity</b> codec only minimized
reconstruction error. Same latent / same bitrate. &check; = the searched concept.</p></div>
<script>
async function load(){{const s=subj.value;const m=(await (await fetch('/api/meta?subj='+s)).json()).metrics;
kpis.innerHTML=`<div class=kpi><div class=v>${{m.ratio}}x</div><div class=k>compression</div></div>`+
`<div class=kpi><div class=v>${{(m.judge.top5*100).toFixed(0)}}%</div><div class=k>uncompressed top5</div></div>`+
`<div class=kpi><div class=v>${{(m.fidelity.top5*100).toFixed(0)}}%</div><div class=k>fidelity top5</div></div>`+
`<div class=kpi><div class="v us">${{(m.actual.top5*100).toFixed(0)}}%</div><div class=k>actual top5</div></div>`+
`<div class=kpi><div class="v us">+${{((m.actual.top5-m.fidelity.top5)*100).toFixed(0)}}pp</div><div class=k>actual lift</div></div>`;go();}}
async function go(){{const d=await (await fetch('/api/search?subj='+subj.value+'&q='+encodeURIComponent(q.value))).json();
if(d.error){{fid.innerHTML=act.innerHTML='<tr><td>'+d.error+'</td></tr>';return;}}
const row=(x,i)=>`<tr class="${{x.concept==d.matched?'hit':''}}"><td class=rank>${{i+1}}</td><td>${{x.concept}}</td><td class=r>${{x.sim.toFixed(3)}}</td></tr>`;
fid.innerHTML=d.fidelity.map(row).join('');act.innerHTML=d.actual.map(row).join('');}}
load();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); qs = urllib.parse.parse_qs(u.query)
        g = lambda k, d="": qs.get(k, [d])[0]
        if u.path == "/":
            self._send(page(), "text/html; charset=utf-8")
        elif u.path == "/api/meta":
            self._send(json.dumps({"metrics": S["subj"][g("subj", S["default"])]["metrics"]}))
        elif u.path == "/api/search":
            subj = g("subj", S["default"]); qf, matched = query_feature(g("q"))
            if qf is None: self._send(json.dumps({"error": f'no CLIP concept matches "{g("q")}"'})); return
            qf = qf.to(S["dev"])
            self._send(json.dumps({"matched": matched,
                                   "fidelity": rank(subj, qf, "fidelity"),
                                   "actual": rank(subj, qf, "actual")}))
        else:
            self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0"); ap.add_argument("--port", type=int, default=8012)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = torch.load(FEAT.format("train"), weights_only=False)
    te = torch.load(FEAT.format("test"), weights_only=False)
    S["text_feats"] = {str(k).lower(): F.normalize(v.float(), dim=0).to(dev)
                       for k, v in {**tr["text_features"], **te["text_features"]}.items()}
    subs = sorted(p.stem for p in ART.glob("sub-*.pt"))
    S["subjects"] = subs; S["default"] = subs[0]; S["dev"] = dev
    S["subj"] = {s: load_subject(s) for s in subs}
    for s in subs:
        for k in S["subj"][s]["emb"]:
            S["subj"][s]["emb"][k] = S["subj"][s]["emb"][k].to(dev)
    print(f"serving NeuroZip-fMRI on http://{args.host}:{args.port}/  subjects={subs}", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
