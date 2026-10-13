#!/usr/bin/env python3
"""
forecast.py — honest baseline forecaster for mandi prices.

Model (deliberately simple, no fake ML):
  For each of the next H days:
    seasonal_h = mean price on that weekday over the last SEASON_WEEKS weeks
    trend_h    = price at anchor + slope * h, slope = OLS fit on last TREND_DAYS
    forecast_h = 0.5 * seasonal_h + 0.5 * trend_h
  Uncertainty band: +/- 1.96 * residual_std, where residual_std is the std of
  this same blended model's one-day-ahead errors on the last TREND_DAYS days.
  (A rough, in-sample band — it UNDERSTATES real risk. Read the caveat.)

The forecast is anchored at the last date present in data/prices.csv
(`as_of`), NOT at today, because the data lags the present by weeks/months.

CLI:
    python forecast.py --crop onion --market Pune [--days 7]
"""
import argparse
import csv
import math
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "prices.csv"

SEASON_WEEKS = 8    # seasonal-naive lookback (same weekday)
TREND_DAYS = 28     # OLS trend window
HORIZON_DEFAULT = 7

CAVEAT = (
    "EXPERIMENTAL baseline (seasonal-naive + linear trend blend). "
    "Mandi prices move on weather, arrivals, policy and speculation — "
    "a 28-day trend + weekly seasonality captures NONE of that. "
    "NOT trading advice. Do not use for real buying/selling decisions."
)


def load_series(crop: str, market: str):
    """Return list of (date, price) sorted, for crop/market (case-insensitive)."""
    if not DATA_PATH.exists():
        raise SystemExit(f"missing {DATA_PATH}; run `python fetch_data.py` first")
    series = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["crop"].lower() == crop.lower() and row["market"].lower() == market.lower():
                d = date.fromisoformat(row["date"])
                series.append((d, float(row["price_per_quintal"])))
    if not series:
        raise SystemExit(f"no data for crop={crop!r} market={market!r} in {DATA_PATH}")
    series.sort()
    return series


def available_series():
    if not DATA_PATH.exists():
        raise SystemExit(f"missing {DATA_PATH}; run `python fetch_data.py` first")
    seen = {}
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["crop"], row["market"])
            seen[key] = seen.get(key, 0) + 1
    return sorted(seen)


def _ols_slope(prices):
    """Slope of least-squares line over prices[0..n-1]; 0.0 if degenerate."""
    n = len(prices)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = (n - 1) / 2
    mean_y = sum(prices) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, prices))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den else 0.0


def forecast(series, horizon=HORIZON_DEFAULT):
    """
    series: list of (date, price) sorted ascending, daily but possibly with gaps.
    Returns dict with as_of, dates, point, lower, upper, residual_std, n_history.
    """
    by_date = dict(series)
    dates = [d for d, _ in series]
    anchor = dates[-1]

    # index of weekdays: map date -> price for seasonal lookup
    seasonal_vals = {}
    for h in range(1, horizon + 1):
        target = anchor + timedelta(days=h)
        wd = target.weekday()
        vals = []
        d = target - timedelta(weeks=1)
        weeks = 0
        while weeks < SEASON_WEEKS and d >= dates[0]:
            if d.weekday() == wd and d in by_date:
                vals.append(by_date[d])
                weeks += 1
            d -= timedelta(weeks=1)
        seasonal_vals[h] = sum(vals) / len(vals) if vals else by_date[anchor]

    window = [p for _, p in series[-TREND_DAYS:]]
    slope = _ols_slope(window)
    anchor_price = window[-1]

    point, lower, upper = [], [], []
    # residual std from one-day-ahead blended errors on the window (in-sample)
    errs = []
    for i in range(1, len(window)):
        # reconstruct seasonal-naive component using the full history before that day
        day = series[-TREND_DAYS + i][0]
        wd = day.weekday()
        past = [p for d, p in series if d < day and d.weekday() == wd][-SEASON_WEEKS:]
        seas = sum(past) / len(past) if past else window[i - 1]
        prev_window = [p for _, p in series][max(0, len(series) - TREND_DAYS - 30):len(series) - TREND_DAYS + i - 1]
        tr = window[i - 1] + _ols_slope(prev_window[-TREND_DAYS:] if len(prev_window) >= 2 else [window[i - 1]]) * 1
        pred = 0.5 * seas + 0.5 * tr
        errs.append(window[i] - pred)
    resid_std = math.sqrt(sum(e * e for e in errs) / max(1, len(errs)))

    out_dates = []
    for h in range(1, horizon + 1):
        trend_h = anchor_price + slope * h
        f = 0.5 * seasonal_vals[h] + 0.5 * trend_h
        band = 1.96 * resid_std
        point.append(f)
        lower.append(max(0.0, f - band))
        upper.append(f + band)
        out_dates.append(anchor + timedelta(days=h))

    return {
        "as_of": anchor,
        "dates": out_dates,
        "point": point,
        "lower": lower,
        "upper": upper,
        "residual_std": resid_std,
        "slope_per_day": slope,
        "n_history": len(series),
        "last_price": by_date[anchor],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline mandi price forecast (experimental).")
    ap.add_argument("--crop", required=False, help="e.g. onion")
    ap.add_argument("--market", required=False, help="e.g. Pune")
    ap.add_argument("--days", type=int, default=HORIZON_DEFAULT)
    ap.add_argument("--list", action="store_true", help="list available crop/market pairs")
    args = ap.parse_args()

    if args.list:
        for crop, market in available_series():
            print(f"{crop} / {market}")
        return 0
    if not args.crop or not args.market:
        raise SystemExit("need --crop and --market (or --list to see options)")
    if not DATA_PATH.exists():
        raise SystemExit(f"missing {DATA_PATH}; run `python fetch_data.py` first")

    series = load_series(args.crop, args.market)
    res = forecast(series, horizon=args.days)

    crop = next(c for c, m in available_series()
                if c.lower() == args.crop.lower() and m.lower() == args.market.lower())
    market = next(m for c, m in available_series()
                  if c.lower() == args.crop.lower() and m.lower() == args.market.lower())

    print(f"crop: {crop}   market: {market}")
    print(f"history: {res['n_history']} daily points, last data date (as_of): {res['as_of']}")
    print(f"last price: Rs {res['last_price']:.0f}/quintal   "
          f"28-day trend slope: {res['slope_per_day']:+.1f} Rs/day")
    print()
    print(f"{'date':<12}{'forecast':>12}{'band':>26}")
    for d, p, lo, hi in zip(res["dates"], res["point"], res["lower"], res["upper"]):
        print(f"{d.isoformat():<12}{p:>11.0f}  [{lo:.0f} .. {hi:.0f}]")
    print()
    print("CAVEAT:", CAVEAT)
    print(f"(band = +/-1.96 x in-sample residual std Rs {res['residual_std']:.0f}; "
          "understates real risk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
