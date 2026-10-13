"""pro-matcher web UI (Flask).

Enter service + area (+ optional min rating) -> ranked pros with per-factor
score breakdown bars.
Run:  python3 app.py        (then open http://127.0.0.1:5001)
"""
from flask import Flask, request, render_template_string

import matcher

app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pro Matcher</title>
<style>
body{font-family:system-ui,sans-serif;max-width:860px;margin:2em auto;padding:0 1em;color:#1a1a1a}
.card{border:1px solid #ddd;border-radius:10px;padding:1em;margin:1em 0}
.top{background:#f6fff6;border-color:#9d9}
.row{display:flex;align-items:center;gap:.6em;margin:.25em 0}
.lbl{width:150px;flex:none}
.bar{flex:1;height:12px;background:#eee;border-radius:6px;overflow:hidden}
.bar>div{height:100%;background:#2a7}
.pts{width:90px;flex:none;text-align:right;font-variant-numeric:tabular-nums}
.total{font-size:1.4em;font-weight:700}
form label{display:block;margin:.6em 0}
select,input,button{font-size:1.05em;padding:.4em .6em}
button{cursor:pointer}
small{color:#555}
.note{background:#fff8e1;border:1px solid #e0c36a;border-radius:8px;padding:.8em;margin:1em 0}
</style></head><body>
<h1>🔧 Pro Matcher <small>— explainable home-service ranking</small></h1>
<div class="note"><b>Seed data:</b> the pros below are fictional placeholders
(phone numbers are obviously fake <code>98XXX-XXXXX</code>). Scores are plain
arithmetic — every factor's contribution is shown. See README for the formula
and honest limitations.</div>
<form method="get" action="/match">
  <label>Service:
    <select name="service">{% for s in services %}
      <option value="{{s}}" {% if s==service %}selected{% endif %}>{{s}}</option>{% endfor %}
    </select></label>
  <label>Area:
    <select name="area">{% for a in areas %}
      <option value="{{a}}" {% if a==area %}selected{% endif %}>{{a}}</option>{% endfor %}
    </select></label>
  <label>Min rating: <input type="number" name="min_rating" min="0" max="5" step="0.1"
    value="{{min_rating}}" style="width:4em"></label>
  <button type="submit">Find pros</button>
</form>
{% if results is not none %}
<h2>Ranked pros ({{results|length}})</h2>
{% for r in results %}
<div class="card {% if loop.first %}top{% endif %}">
  <div class="total">{{loop.index}}. {{r.pro.name}} — {{r.total}}/100</div>
  <small>{{r.pro.id}} · {{", ".join(r.pro.services)}} · {{", ".join(r.pro.areas)}}
  · {{r.pro.jobs_completed}} jobs done · phone {{r.pro.phone}}</small>
  {% for name, pts, label in r.breakdown %}
  <div class="row"><span class="lbl">{{name}}</span>
    <span class="bar"><div style="width:{{(100*pts/weights[name])|round(1)}}%"></div></span>
    <span class="pts">{{pts}}/{{weights[name]|int}}</span></div>
  <small style="margin-left:150px">{{label}}</small>
  {% endfor %}
</div>
{% endfor %}
{% if not results %}<p>No pros match that service / min rating.</p>{% endif %}
{% endif %}
</body></html>"""


@app.route("/")
def index():
    return render_template_string(
        PAGE, services=matcher.all_services(), areas=matcher.all_areas(),
        service="", area="", min_rating=0.0, results=None,
        weights=matcher.WEIGHTS)


@app.route("/match")
def match():
    service = request.args.get("service", "")
    area = request.args.get("area", "")
    try:
        min_rating = float(request.args.get("min_rating", 0) or 0)
    except ValueError:
        min_rating = 0.0
    results = matcher.rank(service, area, min_rating) if service and area else []
    return render_template_string(
        PAGE, services=matcher.all_services(), areas=matcher.all_areas(),
        service=service, area=area, min_rating=min_rating, results=results,
        weights=matcher.WEIGHTS)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
