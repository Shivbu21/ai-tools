"""app.py: minimal web UI for catalog-qa. Single page, text box + Ask button,
answer with cited rows and a mode badge (LLM / extractive / not-found).

Run:  ~/workspace/ai-tools/catalog-venv/bin/python app.py   (serves :8767)
"""
import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

from flask import Flask, jsonify, request

import ask

app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kirana Catalog Q&A</title>
<style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px;background:#faf7f0;color:#222}
h1{font-size:1.4rem} input{width:100%;padding:12px;font-size:1rem;border:2px solid #c9a227;border-radius:8px}
button{margin-top:10px;padding:10px 22px;font-size:1rem;background:#1a5d1a;color:#fff;border:0;border-radius:8px;cursor:pointer}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.8rem;color:#fff;margin:12px 0}
.llm{background:#1a5d1a}.extractive{background:#b26a00}.not-found{background:#a33}
#ans{white-space:pre-wrap;background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px}
ul{padding-left:18px}
</style></head><body>
<h1>🛒 Kirana Catalog Q&amp;A</h1>
<input id="q" placeholder="e.g. what is the price of Tata Salt?" autocomplete="off">
<br><button onclick="go()">Ask</button>
<div id="badge"></div><div id="ans"></div>
<script>
async function go(){
  const q=document.getElementById('q').value;
  document.getElementById('ans').textContent='thinking…';
  const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q})});
  const d=await r.json();
  const b=document.getElementById('badge');
  b.innerHTML='<span class="badge '+d.mode+'">'+d.mode+'</span>';
  let html='<div>'+d.answer.replace(/</g,'&lt;')+'</div>';
  if(d.cited&&d.cited.length){html+='<ul>'+d.cited.slice(0,3).map(c=>'<li>'+c.name.replace(/</g,'&lt;')+' — Rs '+c.price+' per '+c.unit+'</li>').join('')+'</ul>';}
  document.getElementById('ans').innerHTML=html;
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')go();});
</script></body></html>"""


@app.get("/")
def index():
    return PAGE


@app.post("/api/ask")
def api_ask():
    q = (request.get_json(force=True, silent=True) or {}).get("q", "")
    if not q.strip():
        return jsonify({"mode": "not-found", "answer": "Ask me something about the catalog.", "cited": []})
    out = ask.answer(q.strip())
    cited = [{"name": r["name"], "price": r["price"], "unit": r["unit"],
              "category": r["category"]} for r in out.get("cited", [])]
    return jsonify({"mode": out["mode"], "answer": out["answer"], "cited": cited})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8767)
