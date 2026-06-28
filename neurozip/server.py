"""Local NeuroZip UI — a zero-dependency http.server app.

`neurozip serve` opens http://127.0.0.1:7878 where you can drag-and-drop a folder or
point at a directory path and "convert" it. Backed by the beta placeholder in core.py,
so results are randomized until the real codec ships. Localhost-only by default.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, core

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NeuroZip</title><style>
:root{--bg:#0b0e14;--panel:#141a24;--ink:#e6edf3;--muted:#8b97a7;--ok:#3fb950;--ac:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:22px 26px 6px}h1{margin:0;font-size:21px}.tag{color:var(--muted);font-size:13px}
.wrap{padding:18px 26px 30px;max-width:920px}
#drop{border:2px dashed #2a3340;border-radius:14px;padding:38px;text-align:center;color:var(--muted);
background:var(--panel);transition:.15s}#drop.hot{border-color:var(--ac);color:var(--ink);background:#16202e}
.row{display:flex;gap:10px;margin:16px 0;flex-wrap:wrap}
input[type=text]{flex:1;min-width:240px;background:var(--panel);border:1px solid #283142;color:var(--ink);
padding:10px 12px;border-radius:9px}
button{background:var(--ac);color:#04101f;border:0;padding:10px 16px;border-radius:9px;font-weight:600;cursor:pointer}
.ratio{display:flex;gap:9px;align-items:center;color:var(--muted)}
table{width:100%;border-collapse:collapse;margin-top:14px;font-size:13px}
th,td{text-align:left;padding:7px 9px;border-bottom:1px solid #1d2533}th{color:var(--muted);font-weight:500}
.sum{margin-top:14px;padding:12px 14px;background:var(--panel);border:1px solid #232c3a;border-radius:10px}
.sum b{color:var(--ok)}.beta{color:#5b6677;font-size:12px;margin-top:18px}
</style></head><body>
<header><h1>NeuroZip <span style="color:#3a4658;font-size:13px">v__VER__ · beta</span></h1>
<div class="tag">Drag a folder here, or point at a directory path, and convert it.</div></header>
<div class="wrap">
<div id="drop">Drop a folder or files here<br><span style="font-size:12px">(or use the path box below)</span>
<div style="margin-top:12px"><input id="picker" type="file" webkitdirectory multiple style="display:none"/>
<button onclick="document.getElementById('picker').click()">Choose folder…</button></div></div>
<div class="row">
<input id="path" type="text" placeholder="/absolute/path/to/folder"/>
<div class="ratio">ratio <input id="ratio" type="range" min="2" max="200" value="50"/><span id="rv">50×</span></div>
<button onclick="convertPath()">Convert</button></div>
<div id="out"></div>
<div class="beta">Beta placeholder: figures are randomized until the real codec ships. Localhost only.</div>
</div>
<script>
const rng=document.getElementById('ratio'),rv=document.getElementById('rv');
rng.oninput=()=>rv.textContent=rng.value+'×';
const drop=document.getElementById('drop');
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('hot')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('hot')}));
drop.addEventListener('drop',ev=>{const fs=[...ev.dataTransfer.files].map(f=>({name:f.name,size:f.size}));sendFiles(fs)});
document.getElementById('picker').addEventListener('change',ev=>{
  const fs=[...ev.target.files].map(f=>({name:f.webkitRelativePath||f.name,size:f.size}));sendFiles(fs)});
function ratio(){return parseFloat(rng.value)}
async function post(body){const r=await fetch('/convert',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(body)});return r.json()}
async function convertPath(){const p=document.getElementById('path').value.trim();
  if(!p){alert('Enter a folder path or drop files.');return}render(await post({path:p,ratio:ratio()}))}
async function sendFiles(files){if(!files.length)return;render(await post({files,ratio:ratio()}))}
function fmt(n){return n.toLocaleString()}
function render(d){const rows=(d.items||[]).slice(0,200).map(it=>`<tr><td>${it.name}</td>
  <td>${fmt(it.original_bytes)}</td><td>${fmt(it.compressed_bytes)}</td><td>${it.ratio.toFixed(1)}×</td></tr>`).join('');
  document.getElementById('out').innerHTML=`<div class="sum">Converted <b>${d.n_items}</b> item(s):
  ${fmt(d.original_bytes)} B → <b>${fmt(d.compressed_bytes)} B</b> (<b>${d.ratio.toFixed(1)}×</b>)</div>
  <table><thead><tr><th>item</th><th>original</th><th>compressed</th><th>ratio</th></tr></thead><tbody>${rows}</tbody></table>`}
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
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
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/convert":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, json.dumps({"error": "bad request"}))
            return
        self._send(200, json.dumps(_convert(req)))


def _fake_ratio(original: int, ratio: float):
    """Placeholder compressed size jittered around the requested ratio."""
    import random

    comp = max(1, int(original / max(1.0, random.uniform(ratio * 0.8, ratio * 1.15))))
    return comp, original / comp


def _convert(req: dict) -> dict:
    ratio = float(req.get("ratio", 50) or 50)
    items = []
    if req.get("path"):
        for f in core._iter_files(req["path"]):
            ob = core._bytes_of(f) or 1
            comp, r = _fake_ratio(ob, ratio)
            items.append({"name": f, "original_bytes": ob, "compressed_bytes": comp, "ratio": r})
    else:
        for f in req.get("files", []):
            ob = int(f.get("size", 0)) or 1
            comp, r = _fake_ratio(ob, ratio)
            items.append({"name": f.get("name", "item"), "original_bytes": ob,
                          "compressed_bytes": comp, "ratio": r})
    orig = sum(i["original_bytes"] for i in items) or 1
    comp = sum(i["compressed_bytes"] for i in items) or 1
    return {"n_items": len(items), "original_bytes": orig, "compressed_bytes": comp,
            "ratio": orig / comp, "items": items}


def run(host: str = "127.0.0.1", port: int = 7878, open_browser: bool = True) -> int:
    httpd = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    print(f"[neurozip] UI running at {url}  (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[neurozip] stopped.")
    finally:
        httpd.server_close()
    return 0
