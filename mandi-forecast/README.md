# Mandi Price Forecaster

A baseline forecaster for Indian mandi (wholesale market) prices — onion, potato and tomato
across Maharashtra markets (Pune, Vashi, Pimpri, Khadki, Kalyan, Moshi, Narayangaon, Khed(Chakan)).

## What it does

- `fetch_data.py` — reproducible pull of **real** Agmarknet price history into `data/prices.csv`
  (columns: `date, crop, market, price_per_quintal`).
- `forecast.py` — CLI forecaster: seasonal-naive (same weekday, last 8 weeks) + 28-day linear
  trend, 50/50 blended, with a ±1.96σ band from in-sample residuals.
- `evaluate.py` — walk-forward backtest (train strictly before each checkpoint), reports
  MAE/MAPE per crop+market.
- `app.py` — tiny Flask page: pick crop+market → inline-SVG chart of last 180 days of
  history + 7-day forecast band. No JS chart libraries.

## Quick start

```bash
pip install -r requirements.txt
python fetch_data.py                    # pulls real data -> data/prices.csv (~48k rows)
python forecast.py --crop onion --market Pune
python forecast.py --list               # all available crop/market pairs
python evaluate.py                      # walk-forward backtest
python app.py                           # http://127.0.0.1:5000
```

## Data source (real, documented)

`data/prices.csv` = 48,414 rows, 15 crop+market series, 2012-01-01 → 2026-09-28, modal price
Rs/quintal. Full provenance (download URL, sha256, date range) in `data/SOURCE.md`.

What was tried before settling on this source:
1. `agmarknet.gov.in` SearchCmmMkt.aspx — HTTP 403 when fetched programmatically.
2. `data.gov.in` Open Government Data API — needs an API key; keyless requests return nothing.
3. CEDA Ashoka Agmarknet API (`api.ceda.ashoka.edu.in`) — reachable, but needs a free key
   from an email+OTP registration; not pursued (would need the user to sign up).

Fallback used: the public mirror of the CEDA-cleaned Agmarknet extracts, committed as CSV in
`atharvpatil1733-art/farmnex_ai_forecaster` (`data/raw/mandi.csv`) — the same underlying
Agmarknet records, with Agmarknet "Daily Price Arrival Report" exports filling the recent gap.
**Attribution (required by CEDA terms, non-commercial use): data from the Centre for Economic
Data & Analysis (CEDA), Ashoka University, based on Agmarknet.** Charts in this app carry that
credit. See https://ceda.ashoka.edu.in/api-terms-conditions/

## Backtest results (measured, `python evaluate.py`)

Walk-forward: checkpoints every 15 days over the last 180 days of history, 7-day horizon,
model trained strictly on data before each checkpoint. Numbers are the honest output of a
naive baseline, not a tuned model.

| crop   | market      | MAE (Rs/q) | MAPE   |
|--------|-------------|------------|--------|
| Onion  | Pune        | 179        | 10.3%  |
| Onion  | Vashi       | 282        | 12.4%  |
| Onion  | Pimpri      | 321        | 15.8%  |
| Onion  | Khadki      | 61         | 5.0%   |
| Onion  | Kalyan      | 421        | 15.4%  |
| Potato | Pune        | 59         | 5.4%   |
| Potato | Vashi       | 63         | 6.8%   |
| Potato | Khadki      | 38         | 3.8%   |
| Potato | Kalyan      | 165        | 17.3%  |
| Potato | Moshi       | 74         | 6.9%   |
| Tomato | Pune        | 238        | 17.6%  |
| Tomato | Pimpri      | 242        | 13.2%  |
| Tomato | Khadki      | 52         | 5.1%   |
| Tomato | Narayangaon | 243        | 15.7%  |
| Tomato | Khed(Chakan)| 289        | 17.3%  |

Per-crop averages: **Onion MAPE 11.8% · Potato MAPE 8.0% · Tomato MAPE 13.8%.**
Tomato is the hardest (volatile, perishable); potato the easiest (stable staple). Potato's
low error is the baseline being right about "prices barely move", not skill.

## Limitations (read before trusting any number)

- The model knows nothing about weather, arrivals, policy changes, festivals or export bans —
  the actual drivers of mandi prices.
- The ±1.96σ band is computed from **in-sample** residuals; it understates true uncertainty.
- Data lags reality by weeks: forecasts are anchored at the last available data date
  (`as_of`, currently 2026-09-24), not today.

> ⚠️ **Experimental — NOT trading advice.** Do not use for real buying/selling decisions.
> A seasonal-naive + trend baseline is a starting point for learning time series, not a
> market signal.
