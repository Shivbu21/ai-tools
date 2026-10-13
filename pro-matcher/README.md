# Pro Matcher — explainable pro ranking for home services

Enter a service + area (+ optional min rating) → get a ranked list of pros with
**every factor's score contribution shown**. No black box: the score is plain
arithmetic you can audit in `matcher.py`.

## One-command run

```bash
cd pro-matcher
pip install -r requirements.txt
python3 demo.py     # CLI: 4 scripted requests + determinism checks
python3 app.py      # web UI at http://127.0.0.1:5001
```

## Scoring formula (total = 100)

**Gate (not scored):** `service` must be in the pro's service list, otherwise the
pro is *excluded* from the ranking entirely — a plumber never ranks for an
electrician request.

| factor | weight | points |
|---|---|---|
| `area_proximity` | 30 | same area = 30 · nearby area = 18 · far/unknown = 6 (see `NEARBY` map in `matcher.py`) |
| `rating` | 25 | `(rating / 5) × 25` |
| `completion_rate` | 20 | `completion_rate × 20` (0–1 in the data) |
| `availability` | 25 | `(1 − current_workload / 5) × 25` — fewer active jobs scores higher |

**Tie-break (deterministic):** total ↓, then rating ↓, then jobs_completed ↓,
then name ↑. No randomness, no network calls, no ML model — the same input
always produces the same ranking (asserted in `demo.py`, each request ranked
twice; also verified across separate Python processes).

Example — `electrician` in `Noida Sec 62`:
`Ramesh Kumar (E01) = 30.0 + 24.0 + 19.4 + 20.0 = 93.4`
(same area · 4.8★ · 97% completion · 1/5 active jobs).

## Replacing seed data with real pros

`pros.json` is **fictional seed data** (names/stats invented, phones are the
obviously-fake `98XXX-XXXXX` pattern — never call them). To go real, replace it
with a JSON of the same shape:

```json
{"pros": [{"id": "E01", "name": "...", "phone": "+91...", "services": ["electrician"],
  "areas": ["Noida Sec 62"], "rating": 4.8, "jobs_completed": 214,
  "completion_rate": 0.97, "current_workload": 1}]}
```

Then tune `NEARBY` (area adjacency) and `WEIGHTS` in `matcher.py` to your city.

## Honest limitations

- **No real-time availability** — `current_workload` is a static snapshot; a
  "free" pro may be busy. Confirm by phone before promising a slot.
- **No background verification** — ratings/completion rates are self-reported
  seed values; the engine trusts whatever numbers you feed it.
- **No reviews, pricing, or distance** — only area buckets, not km; price is not
  modelled at all.
- Weights are reasonable defaults, not learned from data.

## Files

- `pros.json` — 20 seeded pros (electricians, plumbers, carpenters)
- `matcher.py` — scoring + ranking (importable)
- `demo.py` — CLI demo with determinism asserts
- `app.py` — Flask web UI with breakdown bars
