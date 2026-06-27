"""Local NeuroZip upload UI with real model inference.

`neurozip serve` opens http://127.0.0.1:7878 where you drag-and-drop an EEG epoch
(.npy of shape (63,250)) or pick a bundled demo epoch, choose a compression tier and
variant, and get the real bitrate, ratio, reconstruction error, and a raw-vs-decoded
heatmap rendered by the trained codec. Localhost only.

Needs the `ml` extra (torch + matplotlib) and the checkpoints.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NeuroZip</title><style>
:root{--bg:#0b0e14;--panel:#141a24;--ink:#e6edf3;--mut:#8b97a7;--nz:#3fb950;--fid:#f0883e;--ac:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:22px 26px 4px}h1{margin:0;font-size:21px}.tag{color:var(--mut);font-size:13px}
.wrap{padding:16px 26px 32px;max-width:1000px}
#drop{border:2px dashed #2a3340;border-radius:14px;padding:26px;text-align:center;color:var(--mut);
background:var(--panel);transition:.15s}#drop.hot{border-color:var(--ac);color:var(--ink);background:#16202e}
.row{display:flex;gap:14px;margin:14px 0;flex-wrap:wrap;align-items:center}
select,button{background:var(--panel);border:1px solid #283142;color:var(--ink);padding:9px 12px;border-radius:9px}
button.go{background:var(--ac);color:#04101f;border:0;font-weight:600;cursor:pointer}
.seg{display:flex;border:1px solid #283142;border-radius:9px;overflow:hidden}
.seg button{border:0;border-radius:0;background:transparent}
.seg button.on{background:#1f2a3a;color:#fff}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}
.card{background:var(--panel);border:1px solid #232c3a;border-radius:11px;padding:12px 16px;min-width:120px}
.card .k{color:var(--mut);font-size:12px}.card .v{font-size:22px;font-weight:600;margin-top:2px}
img.fig{max-width:100%;border-radius:10px;border:1px solid #232c3a;margin-top:14px;background:#161922}
table{border-collapse:collapse;margin-top:14px;font-size:13px;width:100%}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid #1d2533}th{color:var(--mut);font-weight:500}
td:first-child,th:first-child{text-align:left}.nz{color:var(--nz)}.fid{color:var(--fid)}
.mut{color:var(--mut);font-size:12px;margin-top:16px}.err{color:#f85149}
</style></head><body>
<header><h1>NeuroZip <span style="color:#3a4658;font-size:13px">v__VER__</span></h1>
<div class="tag">Compress an EEG epoch with the trained task-aware codec and see what survives.</div></header>
<div class="wrap">
  <div id="drop">Drop an EEG epoch <b>.npy</b> (63&times;250) here, or pick a demo epoch below.
    <div style="margin-top:10px"><input id="file" type="file" accept=".npy" style="display:none"/>
    <button onclick="document.getElementById('file').click()">Choose .npy&hellip;</button></div></div>
  <div class="row">
    <label class="tag">demo epoch</label><select id="sample"></select>
    <label class="tag">tier</label><select id="tier"><option>low</option><option>med</option>
      <option selected>high</option><option>xhigh</option></select>
    <label class="tag">variant</label>
    <div class="seg" id="variant"><button data-v="neurozip" class="on">neurozip</button>
      <button data-v="fidelity">fidelity</button></div>
    <button class="go" onclick="run()">Compress</button>
  </div>
  <div id="out"></div>
  <div class="mut">Runs the real codec locally. neurozip = task-aware; fidelity = same bitrate, fidelity-only baseline.</div>
</div>
<script>
let uploaded=null, variant="neurozip";
const seg=document.getElementById('variant');
seg.addEventListener('click',e=>{if(e.target.dataset.v){variant=e.target.dataset.v;
  [...seg.children].forEach(b=>b.classList.toggle('on',b.dataset.v===variant));}});
const drop=document.getElementById('drop');
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hot')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hot')}));
drop.addEventListener('drop',ev=>{ev.preventDefault();readFile(ev.dataTransfer.files[0])});
document.getElementById('file').addEventListener('change',ev=>readFile(ev.target.files[0]));
function readFile(f){if(!f)return;const r=new FileReader();
  r.onload=()=>{uploaded=btoa(String.fromCharCode(...new Uint8Array(r.result)));
    drop.innerHTML='loaded <b>'+f.name+'</b> — will compress your file';};r.readAsArrayBuffer(f);}
async function loadSamples(){const r=await fetch('/api/samples');const d=await r.json();
  const s=document.getElementById('sample');s.innerHTML='';
  (d.concepts||[]).forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=i+': '+c;s.appendChild(o);});}
function card(k,v){return `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`}
async function run(){
  const body={tier:document.getElementById('tier').value,variant:variant};
  if(uploaded) body.npy_b64=uploaded; else body.sample_idx=parseInt(document.getElementById('sample').value);
  document.getElementById('out').innerHTML='<div class="mut">compressing…</div>';
  const r=await fetch('/api/compress',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  if(d.error){document.getElementById('out').innerHTML='<div class="err">'+d.error+'</div>';return;}
  let cmp='';
  if(d.compare){cmp=`<table><thead><tr><th>at ${d.tier} tier</th><th>bpp</th><th>ratio</th><th>MSE</th>
    <th>img@1</th><th>img@5</th><th>txt@1</th></tr></thead><tbody>
    <tr class="nz"><td>neurozip</td><td>${d.compare.neurozip.bpp}</td><td>${d.compare.neurozip.ratio}×</td>
      <td>${d.compare.neurozip.mse}</td><td>${d.compare.neurozip.img1}</td><td>${d.compare.neurozip.img5}</td><td>${d.compare.neurozip.txt1}</td></tr>
    <tr class="fid"><td>fidelity</td><td>${d.compare.fidelity.bpp}</td><td>${d.compare.fidelity.ratio}×</td>
      <td>${d.compare.fidelity.mse}</td><td>${d.compare.fidelity.img1}</td><td>${d.compare.fidelity.img5}</td><td>${d.compare.fidelity.txt1}</td></tr>
    </tbody></table>`;}
  document.getElementById('out').innerHTML=
    `<div class="cards">${card('compression',d.ratio.toFixed(1)+'×')}${card('bits / sample',d.bpp.toFixed(4))}
     ${card('reconstruction MSE',d.mse.toFixed(4))}${card('latent',d.latent_shape.join('×'))}</div>`+
    (d.concept?`<div class="mut">epoch concept: <b>${d.concept}</b></div>`:'')+
    `<img class="fig" src="${d.heatmap}"/>`+cmp;
}
loadSamples();
</script></body></html>"""


class _State:
    ready = False
    codecs = {}
    proj = None
    samples = None
    concepts = None
    scores = {}


def _load_state():
    if _State.ready:
        return
    from . import runtime as rt
    _State.proj = rt.load_projector()
    s = rt.load_samples()
    if s is not None:
        _State.samples, _State.concepts = s
    _State.scores = rt.scores()
    _State.ready = True


def _codec(variant, tier):
    from . import runtime as rt
    key = f"{variant}_{tier}"
    if key not in _State.codecs:
        _State.codecs[key] = rt.load_codec(tier=tier, variant=variant)
    return _State.codecs[key]


def _heatmap(raw, recon, concept):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    vmax = float(np.max(np.abs([raw, recon])))
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    for ax, arr, title in ((axes[0], raw, "raw EEG"),
                           (axes[1], recon, "decompressed")):
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[-200, 996, 63, 0])
        ax.set_title(title, loc="left", color="#e6e8ee", fontsize=11, pad=3)
        ax.set_ylabel("channel", color="#bfc4d2", fontsize=9)
        ax.set_facecolor("#0f1115")
        ax.tick_params(colors="#d0d2d8", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#444a5e")
    axes[1].set_xlabel("time after stimulus (ms)", color="#bfc4d2", fontsize=9)
    fig.patch.set_facecolor("#161922")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#161922")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _compare(tier):
    def row(name):
        s = _State.scores.get(f"{name}_v4_{tier}", {})
        return {"bpp": round(s.get("bpp", 0), 4),
                "ratio": round(s.get("compression_ratio_vs_fp16", 0), 1),
                "mse": round(s.get("mse", 0), 4),
                "img1": round(s.get("retrieval_image", {}).get("top1", 0), 3),
                "img5": round(s.get("retrieval_image", {}).get("top5", 0), 3),
                "txt1": round(s.get("retrieval_text", {}).get("top1", 0), 3)}
    if not _State.scores:
        return None
    return {"neurozip": row("neurozip"), "fidelity": row("fidelity")}


def _do_compress(req: dict) -> dict:
    import numpy as np
    from . import runtime as rt
    _load_state()
    tier = req.get("tier", "high")
    variant = req.get("variant", "neurozip")
    concept = None
    if req.get("npy_b64"):
        raw = np.load(io.BytesIO(base64.b64decode(req["npy_b64"])))
    elif req.get("sample_idx") is not None:
        if _State.samples is None:
            return {"error": "no bundled samples available"}
        i = int(req["sample_idx"])
        raw = _State.samples[i]
        concept = _State.concepts[i]
    else:
        return {"error": "provide a .npy upload or sample_idx"}
    raw = np.asarray(raw, dtype="float32")
    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape != (63, 250):
        return {"error": f"expected EEG shape (63,250), got {raw.shape}"}
    codec = _codec(variant, tier)
    recon, stats, mse = rt.reconstruct(raw, tier=tier, variant=variant, codec=codec)
    return {
        "bpp": stats["bpp"], "ratio": stats["ratio_vs_fp16"], "mse": mse,
        "latent_shape": stats["latent_shape"], "tier": tier, "variant": variant,
        "concept": concept,
        "heatmap": _heatmap(raw, recon[0], concept),
        "compare": _compare(tier),
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.replace("__VER__", __version__), "text/html; charset=utf-8")
        elif self.path == "/api/samples":
            _load_state()
            self._send(200, json.dumps({"concepts": _State.concepts or []}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/compress":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, json.dumps(_do_compress(req)))
        except Exception as e:  # surface model errors to the UI
            self._send(200, json.dumps({"error": f"{type(e).__name__}: {e}"}))


def run(host: str = "127.0.0.1", port: int = 7878, open_browser: bool = True) -> int:
    try:
        from . import runtime as rt
        rt._torch()  # fail fast with a clean message if ml extra is missing
    except ModuleNotFoundError as e:
        print(e)
        return 1
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    print(f"[neurozip] upload UI at {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[neurozip] stopped.")
    finally:
        httpd.server_close()
    return 0
