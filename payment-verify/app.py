"""payment-verify web UI (Flask).

Upload a payment screenshot -> forensic indicator report.
Run:  python3 app.py        (then open http://127.0.0.1:5000)
"""
import base64
import io
import os
import uuid

from flask import Flask, request, render_template_string, abort
from PIL import Image

import forensics

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
ALLOWED = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Payment Screenshot Verifier</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#1a1a1a}
.warn{background:#fff3cd;border:1px solid #e0c36a;padding:1em;border-radius:8px;margin:1em 0}
.score{font-size:2.2em;font-weight:700}
.ind{border:1px solid #ddd;border-radius:8px;padding:1em;margin:1em 0}
.ind h3{margin:0 0 .4em}
.pts{color:#b00;font-weight:700}
img.shot{max-width:100%;border:1px solid #ccc;border-radius:8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1em}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
small{color:#555}
button{font-size:1.1em;padding:.6em 1.2em;cursor:pointer}
</style></head><body>
<h1>🔍 Payment Screenshot Verifier</h1>
<div class="warn"><b>Honest warning:</b> this tool reports <i>indicators</i> of possible
tampering, <b>not proof</b>. A low score does <b>not</b> guarantee authenticity, and a
high score does <b>not</b> prove fraud. Never use this as the sole basis for financial
decisions — always verify the transaction in your own UPI app or bank statement.</div>
<form method="post" action="/analyze" enctype="multipart/form-data">
  <input type="file" name="shot" accept="image/*" required>
  <button type="submit">Analyze</button>
</form>
<p><small>Try the built-in test images:
<a href="/sample/clean">clean_receipt.jpg</a> ·
<a href="/sample/tampered">tampered_receipt.jpg</a></small></p>
</body></html>"""

REPORT = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Analysis report</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#1a1a1a}
.warn{background:#fff3cd;border:1px solid #e0c36a;padding:1em;border-radius:8px;margin:1em 0}
.score{font-size:2.4em;font-weight:700;color:{{color}}}
.ind{border:1px solid #ddd;border-radius:8px;padding:1em;margin:1em 0}
.ind h3{margin:0 0 .4em}
.pts{color:#b00;font-weight:700}
img.shot{max-width:100%;border:1px solid #ccc;border-radius:8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1em}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
small{color:#555}
.bar{height:10px;background:#eee;border-radius:5px;overflow:hidden;margin:.3em 0}
.bar>div{height:100%;background:{{color}}}
</style></head><body>
<h1>Analysis report</h1>
<div class="warn"><b>Honest warning:</b> indicators of possible tampering, <b>not proof</b>.
A low score does <b>not</b> guarantee authenticity. Verify in your own UPI app / bank statement.</div>
<p><small>{{fname}} — {{w}}×{{h}} {{fmt}}</small></p>
<div class="score">Risk score: {{score}} / 100</div>
<div class="bar"><div style="width:{{score}}%"></div></div>
<div class="grid">
  <div><h3>Original</h3><img class="shot" src="data:image/jpeg;base64,{{orig}}"></div>
  <div><h3>ELA heatmap (green box = suspicious region)</h3><img class="shot" src="data:image/png;base64,{{heat}}"></div>
</div>
<h2>Every indicator, with its exact contribution</h2>
{% for i in indicators %}
<div class="ind">
  <h3>{{i.name}} <span class="pts">+{{"%.1f"|format(i.points)}} pts</span>
  <small>(weight {{i.weight}}, risk {{ "%.2f"|format(i.value) }})</small></h3>
  <p><b>Measured:</b> {{i.measured}}</p>
  <p><b>What it means:</b> {{i.meaning}}</p>
</div>
{% endfor %}
<h2>Notes</h2>
<ul>{% for n in notes %}<li>{{n}}</li>{% endfor %}</ul>
<p><a href="/">← analyze another</a></p>
</body></html>"""


def _b64(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _report_for(path: str, fname: str):
    rep = forensics.analyze(path)
    color = "#b00" if rep.risk_score >= 50 else ("#c60" if rep.risk_score >= 25 else "#090")
    orig = Image.open(path)
    thumb = orig.copy()
    thumb.thumbnail((700, 1400))
    heat = rep.ela_heatmap or forensics.ela_heatmap(orig)
    return render_template_string(
        REPORT, fname=fname, w=rep.width, h=rep.height, fmt=rep.fmt,
        score=rep.risk_score, color=color,
        orig=_b64(thumb, "JPEG"), heat=_b64(heat, "PNG"),
        indicators=rep.indicators, notes=rep.notes)


@app.route("/")
def index():
    return PAGE


@app.route("/sample/<which>")
def sample(which):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    path = os.path.join(base, {"clean": "clean_receipt.jpg",
                               "tampered": "tampered_receipt.jpg"}.get(which, ""))
    if not os.path.isfile(path):
        abort(404)
    return _report_for(path, os.path.basename(path))


@app.route("/analyze", methods=["POST"])
def analyze():
    f = request.files.get("shot")
    if not f or not f.filename:
        abort(400, "no file uploaded")
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        abort(400, f"unsupported file type {ext}")
    tmp = f"/tmp/pv_{uuid.uuid4().hex}{ext}"
    f.save(tmp)
    try:
        Image.open(tmp).verify()  # reject non-images early
        return _report_for(tmp, f.filename)
    except Exception as e:
        abort(400, f"could not read image: {e}")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
