#!/usr/bin/env python3
"""
app.py — resume ranker web UI. Flask.

Upload multiple resume PDFs + paste a job description -> ranked table with
per-candidate explanations (matched/missing skills, shared keywords).

Run:  python app.py            # http://127.0.0.1:5001
"""
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

from ranker import rank_resumes

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB total
MAX_FILES = 20

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Resume ranker</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#f6f4ee;color:#222;
     max-width:900px;margin:0 auto;padding:24px}
h1{font-size:1.5rem;margin:0 0 4px}
.sub{color:#6b6558;margin:0 0 16px;font-size:.95rem}
form{background:#fff;border:1px solid #e2dccb;border-radius:10px;padding:16px;margin-bottom:20px}
textarea{width:100%;min-height:140px;font:inherit;padding:10px;border-radius:8px;
         border:1px solid #cfc8b8;box-sizing:border-box}
input[type=file]{margin:10px 0}
button{background:#1f6feb;color:#fff;border:none;border-radius:8px;
       padding:10px 18px;font-size:1rem;cursor:pointer}
.card{background:#fff;border:1px solid #e2dccb;border-radius:10px;padding:14px 16px;margin:12px 0}
.rank{font-size:1.05rem;font-weight:bold}
.score{font-size:1.4rem;font-weight:bold;color:#1f6feb}
.bar{height:8px;background:#eee;border-radius:4px;margin:6px 0 10px}
.bar>div{height:8px;background:#1f6feb;border-radius:4px}
.chip{display:inline-block;background:#eef4ff;border:1px solid #c9dcff;border-radius:20px;
      padding:2px 10px;margin:2px;font-size:.85rem}
.chip.miss{background:#fff1f0;border-color:#f3c1bd}
.kw{font-family:ui-monospace,monospace;background:#f3f1e8;border-radius:4px;
    padding:1px 6px;margin:1px;font-size:.82rem;display:inline-block}
.meta{color:#6b6558;font-size:.85rem}
.note{background:#fff7e6;border:1px solid #e3c878;border-radius:8px;padding:10px;
      font-size:.88rem;margin-top:8px}
h3{margin:14px 0 6px;font-size:.95rem;color:#444}
.warn{font-size:.85rem;color:#6b6558;margin-top:18px}
</style></head><body>
<h1>📄 Resume ranker</h1>
<p class="sub">Keyword + skill-overlap ranking for small-business hiring. The explanation is the product.</p>
<form method="post" enctype="multipart/form-data">
  <label><b>Job description</b></label><br>
  <textarea name="jd" placeholder="Paste the job description here...">{{ jd }}</textarea><br>
  <label><b>Resume PDFs</b> (up to {{ max_files }})</label><br>
  <input type="file" name="resumes" accept=".pdf" multiple required><br>
  <button type="submit">Rank resumes</button>
</form>
{{ body | safe }}
<p class="warn">Honest limits: this is keyword/phrase matching + TF-IDF similarity, not semantic
understanding — it can't judge quality of experience or read between the lines.
Scanned/image PDFs have no extractable text (OCR not included) and always score 0;
check those manually. Scores compare candidates <i>against each other for this JD</i>,
not against an absolute bar.</p>
</body></html>"""


def render_results(results):
    if not results:
        return "<p>No resumes uploaded.</p>"
    out = [f"<h2>Ranked: {len(results)} candidates</h2>"]
    for i, r in enumerate(results, 1):
        e = r["explanation"]
        w = min(100, max(0, r["score"]))
        matched = "".join(f'<span class="chip">{s}</span>' for s in e["matched_skills"]) or "—"
        missing = "".join(f'<span class="chip miss">{s}</span>' for s in e["missing_skills"]) or "—"
        other = "".join(f'<span class="chip">{s}</span>' for s in e["other_skills"]) or "—"
        kws = "".join(f'<span class="kw">{k}</span>' for k in e["shared_keywords"]) or "—"
        note = f'<div class="note">{e["note"]}</div>' if e["note"] else ""
        out.append(f"""<div class="card">
  <div class="rank">#{i} — {r['name']}</div>
  <div class="score">{r['score']:.1f}<span class="meta"> / 100</span></div>
  <div class="bar"><div style="width:{w:.0f}%"></div></div>
  <div class="meta">skill overlap {r['skill_score']:.2f} · JD-keyword similarity {r['tfidf_cosine']:.3f} ·
  text via {e['extract_method']}</div>
  <h3>✅ Matched skills ({len(e['matched_skills'])})</h3><div>{matched}</div>
  <h3>❌ Missing from resume — required by JD ({len(e['missing_skills'])})</h3><div>{missing}</div>
  <h3>➕ Other skills found in resume</h3><div>{other}</div>
  <h3>🔑 Shared JD keywords ({e['n_shared_keywords']})</h3><div>{kws}</div>
  {note}
</div>""")
    return "\n".join(out)


@app.route("/", methods=["GET", "POST"])
def index():
    body, jd = "", ""
    if request.method == "POST":
        jd = request.form.get("jd", "")
        files = request.files.getlist("resumes")[:MAX_FILES]
        resumes = []
        for f in files:
            name = secure_filename(f.filename or "unnamed.pdf")
            if not name.lower().endswith(".pdf"):
                continue
            resumes.append({"name": name, "pdf_bytes": f.read()})
        if not jd.strip():
            body = "<p><b>Paste a job description first.</b></p>"
        elif not resumes:
            body = "<p><b>Upload at least one PDF resume.</b></p>"
        else:
            body = render_results(rank_resumes(jd, resumes))
    return render_template_string(PAGE, body=body, jd=jd, max_files=MAX_FILES)


if __name__ == "__main__":
    app.run(port=5001, debug=False)
