#!/usr/bin/env python3
"""
fetch_data.py — reproducible pull of real mandi (wholesale market) price history.

DATA SOURCE STORY (documented honestly):
  1. Tried agmarknet.gov.in SearchCmmMkt.aspx directly        -> HTTP 403 (blocked)
  2. Tried data.gov.in Open Government Data API without key  -> empty response
     (the API requires a free api-key the builder does not have)
  3. Tried the CEDA Ashoka Agmarknet API (api.ceda.ashoka.edu.in)
     -> reachable, but every endpoint needs a free API key obtained via an
        email+OTP registration flow; registering an account in the user's name
        was out of scope for this build, so this path was not pursued.
  4. FALLBACK USED: the public mirror of the exact same CEDA-cleaned Agmarknet
     data, committed as a CSV in a public research repo:
       https://raw.githubusercontent.com/atharvpatil1733-art/farmnex_ai_forecaster/main/data/raw/mandi.csv
     That CSV's own provenance (documented in that repo's data/README.md):
       - Centre for Economic Data & Analysis (CEDA), Ashoka University,
         cleaned extracts of Agmarknet (agmarknet.gov.in) daily mandi records,
         plus agmarknet.gov.in "Daily Price Arrival Report" CSV exports for
         2025-11-07 .. 2026-09-28 to fill CEDA's lag.
       - Free for non-commercial use; charts/apps must credit CEDA
         (https://ceda.ashoka.edu.in/api-terms-conditions/).

This script downloads that CSV, keeps only rows with a modal price for the
target crops/markets, and writes data/prices.csv with columns:
    date, crop, market, price_per_quintal
It also writes data/SOURCE.md recording the exact URL, byte size, sha256,
date range and row counts so the provenance is auditable.

Usage:
    python fetch_data.py            # full pull (default)
    python fetch_data.py --check    # only verify an existing data/prices.csv
"""
import argparse
import csv
import hashlib
import sys
from pathlib import Path
from urllib.request import Request, urlopen

UPSTREAM_URL = (
    "https://raw.githubusercontent.com/atharvpatil1733-art/farmnex_ai_forecaster"
    "/main/data/raw/mandi.csv"
)
UPSTREAM_DESCRIPTION = (
    "CEDA (Ashoka University) cleaned Agmarknet mandi records + agmarknet.gov.in "
    "'Daily Price Arrival Report' exports; mirrored at "
    "https://github.com/atharvpatil1733-art/farmnex_ai_forecaster (data/raw/mandi.csv)"
)

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "mandi_upstream.csv"
OUT_PATH = ROOT / "data" / "prices.csv"
SOURCE_MD = ROOT / "data" / "SOURCE.md"

# crop -> markets to keep (chosen for long, dense histories in the upstream file)
TARGETS = {
    "Onion":  ["Pune", "Vashi", "Pimpri", "Khadki", "Kalyan"],
    "Potato": ["Vashi", "Pune", "Khadki", "Kalyan", "Moshi"],
    "Tomato": ["Pune", "Pimpri", "Khadki", "Narayangaon", "Khed(Chakan)"],
}

MIN_ROWS_PER_SERIES = 365  # require at least ~1 year of history per crop+market


def download() -> bytes:
    req = Request(UPSTREAM_URL, headers={"User-Agent": "mandi-forecast/1.0 (research)"})
    with urlopen(req, timeout=180) as resp:
        if resp.status != 200:
            raise RuntimeError(f"download failed: HTTP {resp.status}")
        return resp.read()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(rows_in: Path, rows_out: Path) -> dict:
    kept = []
    dropped_series = []
    counts = {}
    with open(rows_in, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop = row["commodity"].strip()
            market = row["market"].strip()
            modal = row["modal_price"].strip()
            if crop not in TARGETS or market not in TARGETS[crop] or not modal:
                continue
            try:
                price = float(modal)
            except ValueError:
                continue
            if price <= 0:
                continue
            date = row["date"].strip()  # already YYYY-MM-DD upstream
            kept.append((date, crop, market, price))
            counts[(crop, market)] = counts.get((crop, market), 0) + 1

    # drop thin series so every series has >= 1 year of history
    final = [r for r in kept if counts[(r[1], r[2])] >= MIN_ROWS_PER_SERIES]
    for key, n in sorted(counts.items()):
        if n < MIN_ROWS_PER_SERIES:
            dropped_series.append((key, n))

    final.sort(key=lambda r: (r[1], r[2], r[0]))
    with open(rows_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "crop", "market", "price_per_quintal"])
        for date, crop, market, price in final:
            w.writerow([date, crop, market, f"{price:.2f}"])
    return {
        "rows": len(final),
        "series": sorted({(r[1], r[2]) for r in final}),
        "dropped": dropped_series,
        "date_min": min(r[0] for r in final),
        "date_max": max(r[0] for r in final),
    }


def write_source_md(stats: dict, digest: str, size: int) -> None:
    lines = [
        "# data/prices.csv — provenance",
        "",
        f"- Upstream file: `{UPSTREAM_URL}`",
        f"- Upstream description: {UPSTREAM_DESCRIPTION}",
        f"- Downloaded: sha256 `{digest}`, {size:,} bytes",
        f"- Built: {stats['rows']:,} rows, {len(stats['series'])} crop+market series",
        f"- Date range: {stats['date_min']} .. {stats['date_max']}",
        "- Prices: modal (most-traded) price, Rs/quintal, exactly as reported by Agmarknet.",
        "- Series kept (all >= 365 rows of history):",
    ]
    for crop, market in stats["series"]:
        lines.append(f"  - {crop} / {market}")
    if stats["dropped"]:
        lines.append("- Series dropped for thin history (< 365 rows):")
        for (crop, market), n in stats["dropped"]:
            lines.append(f"  - {crop} / {market} ({n} rows)")
    lines += [
        "",
        "## Attribution (required by CEDA terms, non-commercial use)",
        "Data: Centre for Economic Data & Analysis (CEDA), Ashoka University,",
        "based on Agmarknet (agmarknet.gov.in).",
        "See https://ceda.ashoka.edu.in/api-terms-conditions/",
        "",
        "## What was tried before this fallback",
        "- agmarknet.gov.in SearchCmmMkt.aspx: HTTP 403 when fetched programmatically.",
        "- data.gov.in API without an api-key: empty response (key required, none available).",
        "- CEDA Ashoka API (api.ceda.ashoka.edu.in): reachable but requires a free API key",
        "  from an email+OTP registration flow; not pursued (would need the user to sign up).",
    ]
    SOURCE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="only verify an existing data/prices.csv")
    args = ap.parse_args()

    if args.check:
        if not OUT_PATH.exists():
            print(f"MISSING: {OUT_PATH} — run `python fetch_data.py` first", file=sys.stderr)
            return 1
        print(f"OK: {OUT_PATH} exists")
        return 0

    print(f"downloading {UPSTREAM_URL} ...")
    if RAW_PATH.exists() and RAW_PATH.stat().st_size > 1_000_000:
        blob = RAW_PATH.read_bytes()
        print(f"reusing existing {RAW_PATH} ({len(blob):,} bytes)")
    else:
        blob = download()
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_bytes(blob)
        print(f"saved {len(blob):,} bytes -> {RAW_PATH}")
    digest = sha256_of(RAW_PATH)
    print(f"sha256 {digest[:16]}...")

    stats = build(RAW_PATH, OUT_PATH)
    write_source_md(stats, digest, len(blob))
    print(f"wrote {stats['rows']:,} rows -> {OUT_PATH}")
    print(f"series: {len(stats['series'])}, range {stats['date_min']}..{stats['date_max']}")
    if stats["dropped"]:
        print(f"dropped thin series: {stats['dropped']}")
    print(f"provenance -> {SOURCE_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
