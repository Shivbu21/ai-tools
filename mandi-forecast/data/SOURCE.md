# data/prices.csv — provenance

- Upstream file: `https://raw.githubusercontent.com/atharvpatil1733-art/farmnex_ai_forecaster/main/data/raw/mandi.csv`
- Upstream description: CEDA (Ashoka University) cleaned Agmarknet mandi records + agmarknet.gov.in 'Daily Price Arrival Report' exports; mirrored at https://github.com/atharvpatil1733-art/farmnex_ai_forecaster (data/raw/mandi.csv)
- Downloaded: sha256 `d41db48f00614e6af01ac766f7c1c707ca3d625d89dade11dce4074cb7ed83c8`, 4,329,587 bytes
- Built: 48,414 rows, 15 crop+market series
- Date range: 2012-01-01 .. 2026-09-28
- Prices: modal (most-traded) price, Rs/quintal, exactly as reported by Agmarknet.
- Series kept (all >= 365 rows of history):
  - Onion / Kalyan
  - Onion / Khadki
  - Onion / Pimpri
  - Onion / Pune
  - Onion / Vashi
  - Potato / Kalyan
  - Potato / Khadki
  - Potato / Moshi
  - Potato / Pune
  - Potato / Vashi
  - Tomato / Khadki
  - Tomato / Khed(Chakan)
  - Tomato / Narayangaon
  - Tomato / Pimpri
  - Tomato / Pune

## Attribution (required by CEDA terms, non-commercial use)
Data: Centre for Economic Data & Analysis (CEDA), Ashoka University,
based on Agmarknet (agmarknet.gov.in).
See https://ceda.ashoka.edu.in/api-terms-conditions/

## What was tried before this fallback
- agmarknet.gov.in SearchCmmMkt.aspx: HTTP 403 when fetched programmatically.
- data.gov.in API without an api-key: empty response (key required, none available).
- CEDA Ashoka API (api.ceda.ashoka.edu.in): reachable but requires a free API key
  from an email+OTP registration flow; not pursued (would need the user to sign up).
