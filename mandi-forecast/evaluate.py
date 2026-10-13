#!/usr/bin/env python3
"""
evaluate.py — walk-forward backtest of the forecast.py baseline.

Method (no lookahead):
  For each crop+market series, place checkpoints every STEP days over the last
  TEST_SPAN days of history. At each checkpoint, train ONLY on data before the
  checkpoint, forecast HORIZON days ahead, and compare against actuals.
  Reports MAE and MAPE per series, and averages per crop.

Usage:
    python evaluate.py [--test-span 180] [--step 15] [--horizon 7]
Prints a table; the numbers go into README.md by hand (they are real).
"""
import argparse
import statistics
import sys
from datetime import timedelta

from forecast import available_series, forecast, load_series


def backtest_series(crop, market, test_span=180, step=15, horizon=7):
    series = load_series(crop, market)
    last = series[-1][0]
    start = last - timedelta(days=test_span)
    by_date = dict(series)

    maes, mapes, n_points = [], [], 0
    ckpt = start
    while ckpt <= last - timedelta(days=horizon):
        hist = [(d, p) for d, p in series if d <= ckpt]
        if len(hist) < 60:  # need minimum history to fit
            ckpt += timedelta(days=step)
            continue
        res = forecast(hist, horizon=horizon)
        errs, apes = [], []
        for d, pred in zip(res["dates"], res["point"]):
            if d in by_date:
                actual = by_date[d]
                errs.append(abs(actual - pred))
                if actual > 0:
                    apes.append(abs(actual - pred) / actual)
        if errs:
            maes.append(sum(errs) / len(errs))
            n_points += len(errs)
        if apes:
            mapes.append(sum(apes) / len(apes))
        ckpt += timedelta(days=step)

    return {
        "crop": crop, "market": market,
        "checkpoints": len(maes),
        "mae": statistics.mean(maes) if maes else float("nan"),
        "mape_pct": 100 * statistics.mean(mapes) if mapes else float("nan"),
        "n_points": n_points,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-span", type=int, default=180)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()

    print(f"walk-forward backtest: checkpoints every {args.step}d over last "
          f"{args.test_span}d, horizon {args.horizon}d, train strictly < checkpoint")
    print()
    results = []
    for crop, market in available_series():
        r = backtest_series(crop, market, args.test_span, args.step, args.horizon)
        results.append(r)
        print(f"{crop:8s} {market:14s} ckpts={r['checkpoints']:3d} "
              f"MAE=Rs {r['mae']:7.0f}/q  MAPE={r['mape_pct']:5.1f}%  (n={r['n_points']})")

    print()
    print("per-crop averages (mean of series MAE/MAPE):")
    for crop in sorted({r["crop"] for r in results}):
        sub = [r for r in results if r["crop"] == crop]
        mae = statistics.mean(r["mae"] for r in sub)
        mape = statistics.mean(r["mape_pct"] for r in sub)
        print(f"  {crop:8s} MAE=Rs {mae:7.0f}/q   MAPE={mape:5.1f}%   ({len(sub)} markets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
