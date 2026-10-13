# Resume Ranker

Rank job applicants for small-business hiring: upload resume PDFs + paste a job
description → ranked table where **every candidate gets a plain-English explanation**
(matched skills, missing skills, shared JD keywords). The explanation is the product.

## Quick start

```bash
pip install -r requirements.txt
python test_ranking.py     # one-command demo: builds 5 sample resumes, ranks them
                           # against 2 sample JDs, prints tables, asserts the
                           # best-fit resume ranks #1 for each role
python app.py              # web UI at http://127.0.0.1:5001
```

## How scoring works (documented, deterministic)

1. **Text extraction** — `pdfplumber` first; if it finds nothing, `pdfminer.six`;
   if still nothing, the resume is flagged *"no extractable text"* and scores 0
   (never crashes).
2. **Skill extraction** — phrase matching against the curated `skills.txt`
   (~130 phrases: accounting, office, sales, trades, driving/delivery, food,
   beauty, tailoring, digital/tech, spoken languages). Case-insensitive,
   word-boundary, optional plural `s`.
3. **Relevance** — TF-IDF cosine similarity between the JD and each resume
   (scikit-learn, english stop words, 1–2 grams).
4. **Final score** = `100 × (0.5 × skill_overlap + 0.5 × tfidf_cosine)`,
   where `skill_overlap = |resume skills ∩ JD skills| / |JD skills|`.
   If the JD contains no recognized skills, the skill component is 0 and the
   ranking leans on keyword similarity (noted in the explanation).

Per-candidate explanation shows: matched skills, missing skills (required by the
JD but absent from the resume), other skills found, top shared JD↔resume keywords
with the count, and the extraction method used.

## Measured demo results (`python test_ranking.py`)

Accountant JD (Tally Prime, GST filing, Excel, 2+ yrs):

| # | resume | score | why |
|---|--------|-------|-----|
| 1 | priya_sharma_accountant.pdf | 70.2 | 16/16 JD skills matched (Tally Prime, GST filing, GSTR-1/3B…) |
| 2 | rahul_verma_fresher.pdf | 11.1 | B.Com + Excel, but no Tally/GST |
| 3 | neha_gupta_assistant.pdf | 7.5 | office skills only |
| 4 | suresh_yadav_driver.pdf | 4.5 | wrong field entirely |
| 5 | amit_kumar_electrician.pdf | 0.5 | wrong field entirely |

Delivery-executive JD (two-wheeler licence, delivery experience):

| # | resume | score | why |
|---|--------|-------|-----|
| 1 | suresh_yadav_driver.pdf | 71.8 | 7/7 JD skills (two-wheeler + LMV licence, route planning…) |
| 2 | neha_gupta_assistant.pdf | 8.9 | mentions delivery in her objective — honestly picked up |
| 3–5 | electrician / fresher / accountant | ~1 | no relevant skills |

The best-fit resume ranks #1 for both roles with a ~60-point margin — asserted in
`test_ranking.py`, which fails loudly if ranking ever stops discriminating.

## Honest limitations

- **Keyword-based, not semantic understanding.** It matches phrases; it cannot judge
  the *quality* of experience, verify claims, or read between the lines. A resume that
  stuffs keywords will score well — always interview before hiring.
- **Image/scanned PDFs need OCR, which is not included.** They are reported as
  "no extractable text" and score 0 — check those manually instead of rejecting them.
- **Scores are relative, not absolute.** 70 vs 11 means "much better fit *for this JD*",
  not "70% qualified". A weak applicant pool still produces a #1.
- `skills.txt` is hand-curated for Indian small-business roles; extend it for other
  domains. Matching is intentionally literal — add a phrase only if you'd screen for it.
- No personal data leaves your machine: everything runs locally.
