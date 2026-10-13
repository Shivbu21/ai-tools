#!/usr/bin/env python3
"""
app.py — tiny web view for mandi-forecast. Flask, no JS chart libs: inline SVG.

Run:  python app.py            # http://127.0.0.1:5000
"""
import math
from datetime import timedelta
from flask import Flask, request, render_template_string

from forecast import CAVEAT, available_series, forecast, load_series

app = Flask(__name__)

CREDIT = ("Data: CEDA (Ashoka University) Agmarknet records — "
          "https://ceda.ashoka.edu.in/api-terms-conditions/")


def svg_chart(series, fc, width=860, height=340):
    """Inline SVG: last HISTORY_DAYS of history + 7-day forecast band."""
    hist = series[-HISTORY_DAYS:]
    hx = [d for d, _ in hist]
    all_dates = hx + list(fc["dates"])
    prices = [p for _, p in hist] + list(fc["point"]) + list(fc["lower"]) + list(fc["upper"])
    lo, hi = min(prices), max(prices)
    pad = (hi - lo) * 0.08 or 1.0
    lo, hi = lo - pad, hi + pad

    def X(i):
        return 46 + i * (width - 66) / max(1, len(all_dates) - 1)

    def Y(v):
        return height - 34 - (v - lo) / (hi - lo) * (height - 60)

    n_hist = len(hx)
    hist_pts = " ".join(f"{X(i):.1f},{Y(p):.1f}" for i, (_, p) in enumerate(hist))
    fc_pts = " ".join(f"{X(n_hist + i):.1f},{Y(p):.1f}"
                      for i, p in enumerate(fc["point"]))
    # band polygon: uppers forward, lowers backward
    band = (" ".join(f"{X(n_hist + i):.1f},{Y(v):.1f}" for i, v in enumerate(fc["upper"])) +
            " " + " ".join(f"{X(n_hist + i):.1f},{Y(v):.1f}"
                           for i, v in reversed(list(enumerate(fc["lower"])))))

    # y-axis ticks
    ticks = ""
    for k in range(5):
        v = lo + pad + (hi - lo - 2 * pad) * k / 4
        y = Y(v)
        ticks += (f'<line x1="46" y1="{y:.1f}" x2="{width-20}" y2="{y:.1f}" '
                  f'stroke="#e5e1d8" stroke-width="1"/>'
                  f'<text x="40" y="{y+4:.1f}" font-size="10" text-anchor="end" '
                  f'fill="#6b6558">Rs {v:,.0f}</text>')

    # anchor divider line (history ends / forecast begins)
    ax = X(n_hist - 1)
    label = (f'<text x="{ax:.1f}" y="18" font-size="11" text-anchor="middle" '
             f'fill="#6b6558">as of {fc["as_of"]} → 7-day forecast</text>')

    return f"""<svg viewBox="0 0 {width} {height}" width="100%" role="img"
     aria-label="price history and forecast chart">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#fbf9f4" rx="8"/>
  {ticks}
  <line x1="{ax:.1f}" y1="26" x2="{ax:.1f}" y2="{height-34}"
        stroke="#6b6558" stroke-dasharray="5 4" stroke-width="1.2"/>
  <polygon points="{band}" fill="#2f7d4f" opacity="0.18"/>
  <polyline points="{hist_pts}" fill="none" stroke="#1f2937"
            stroke-width="2" stroke-linejoin="round"/>
  <polyline points="{fc_pts}" fill="none" stroke="#2f7d4f"
            stroke-width="2.5" stroke-dasharray="7 4" stroke-linejoin="round"/>
  <text x="{X(len(all_dates)-1):.1f}" y="{Y(fc['point'][-1])-10:.1f}" font-size="11"
        text-anchor="end" fill="#2f7d4f" font-weight="bold">
    Rs {fc['point'][-1]:,.0f}/q day+7</text>
  {label}
  <text x="{width-20}" y="{height-10}" font-size="9" text-anchor="end" fill="#8a8474">
    {CREDIT}</text>
</svg>"""


HISTORY_DAYS = 180

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mandi price forecaster</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#f4f1e9;color:#222;
     max-width:960px;margin:0 auto;padding:24px}
h1{font-size:1.5rem;margin:0 0 4px}
.sub{color:#6b6558;margin:0 0 16px;font-size:.95rem}
form{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}
select,button{font-size:1rem;padding:8px 10px;border-radius:8px;border:1px solid #cfc8b8}
button{background:#2f7d4f;color:#fff;border:none;cursor:pointer}
table{border-collapse:collapse;margin:16px 0;font-size:.92rem}
th,td{border:1px solid #ddd5c2;padding:6px 10px;text-align:right}
th{background:#efe9da}td:first-child,th:first-child{text-align:left}
.warn{background:#fff7e6;border:1px solid #e3c878;border-radius:8px;padding:12px;
      font-size:.9rem;margin-top:16px}
.credit{font-size:.8rem;color:#8a8474;margin-top:14px}
</style></head><body>
<h1>🌾 Mandi price forecaster</h1>
<p class="sub">Baseline forecast (seasonal-naive + linear trend) on real Agmarknet history.</p>
<form method="get">
<select name="crop">{{ crop_opts }}</select>
<select name="market">{{ market_opts }}</select>
<button type="submit">Forecast</button>
</form>
{{ body | safe }}
<div class="warn"><b>Experimental — NOT trading advice.</b> {{ caveat }}</div>
<p class="credit">{{ credit }}</p>
</body></html>"""


@app.route("/")
def index():
    crops = sorted({c for c, _ in available_series()})
    markets = sorted({m for _, m in available_series()})
    crop = request.args.get("crop", "Onion")
    market = request.args.get("market", "Pune")

    def opts(items, sel):
        return "".join(
            f'<option value="{i}"{" selected" if i == sel else ""}>{i}</option>'
            for i in items)

    try:
        series = load_series(crop, market)
    except SystemExit:
        body = f"<p>No data for {crop} / {market}. Try another pair.</p>"
        return render_template_string(
            PAGE, crop_opts=opts(crops, crop), market_opts=opts(markets, market),
            body=body, caveat=CAVEAT, credit=CREDIT)

    fc = forecast(series, horizon=7)
    chart = svg_chart(series, fc)
    rows = "".join(
        f"<tr><td>{d}</td><td>Rs {p:,.0f}</td>"
        f"<td>Rs {lo:,.0f} – Rs {hi:,.0f}</td></tr>"
        for d, p, lo, hi in zip(fc["dates"], fc["point"], fc["lower"], fc["upper"]))
    body = (f"{chart}<table><tr><th>date</th><th>forecast /quintal</th>"
            f"<th>band (±1.96σ, in-sample)</th></tr>{rows}</table>"
            f"<p class='sub'>History: {len(series)} daily points · last data date "
            f"{fc['as_of']} · last price Rs {fc['last_price']:,.0f}/quintal.</p>")
    return render_template_string(
        PAGE, crop_opts=opts(crops, crop), market_opts=opts(markets, market),
        body=body, caveat=CAVEAT, credit=CREDIT)


if __name__ == "__main__":
    app.run(debug=False)
