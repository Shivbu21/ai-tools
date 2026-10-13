#!/usr/bin/env python3
"""
ranker.py — resume ranking library (no ML magic, just honest keyword statistics).

Pipeline per resume:
  1. extract_text(): pdfplumber first; if nothing, pdfminer.six; if still
     nothing -> marked "no extractable text" (scanned/image PDFs need OCR,
     which is NOT included).
  2. extract_skills(): phrase matching against skills.txt (case-insensitive,
     word boundaries, optional plural 's').
  3. relevance(): TF-IDF cosine similarity between the job description and the
     resume (scikit-learn, english stop words, 1-2 grams).
  4. score = 100 * (0.5 * skill_score + 0.5 * tfidf_cosine), where
     skill_score = |resume skills ∩ JD skills| / |JD skills|
     (0 if the JD contains no listed skills — then ranking leans on TF-IDF).
  5. explain(): matched skills, missing skills (in JD, not in resume),
     other resume skills, and top shared JD↔resume keywords.

Everything is deterministic and inspectable: the explanation IS the product.
"""
import io
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
SKILLS_PATH = ROOT / "skills.txt"

W_SKILL = 0.5   # weight of skill-overlap component
W_TFIDF = 0.5   # weight of TF-IDF relevance component


def load_skills(path: Path = SKILLS_PATH):
    skills = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            skills.append(line.lower())
    # longest phrases first so "tally prime" wins over "tally" in display
    skills.sort(key=len, reverse=True)
    return skills


SKILLS = load_skills()
_SKILL_RES = {s: re.compile(r"\b" + re.escape(s) + r"s?\b", re.IGNORECASE)
              for s in SKILLS}


def extract_text(pdf_bytes: bytes):
    """
    Returns (text, method) where method is one of
    'pdfplumber', 'pdfminer' or 'none' (no extractable text).
    Never raises on a corrupt/empty file — returns ('', 'none').
    """
    if not pdf_bytes:
        return "", "none"
    # 1) pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        if text.strip():
            return text, "pdfplumber"
    except Exception:
        pass
    # 2) pdfminer.six fallback
    try:
        from pdfminer.high_level import extract_text as pm_extract
        text = pm_extract(io.BytesIO(pdf_bytes)) or ""
        if text.strip():
            return text, "pdfminer"
    except Exception:
        pass
    return "", "none"


def extract_skills(text: str):
    """Set of skill phrases from skills.txt found in text."""
    found = set()
    for skill, rx in _SKILL_RES.items():
        if rx.search(text):
            found.add(skill)
    return found


def rank_resumes(jd_text: str, resumes):
    """
    jd_text: job description string.
    resumes: list of dicts {"name": str, "pdf_bytes": bytes}.
    Returns list of result dicts sorted by score desc, each with an
    'explanation' dict. Deterministic.
    """
    jd_text = jd_text or ""
    jd_skills = extract_skills(jd_text)

    parsed = []
    for r in resumes:
        text, method = extract_text(r["pdf_bytes"])
        parsed.append({
            "name": r["name"],
            "text": text,
            "method": method,
            "skills": extract_skills(text) if text.strip() else set(),
        })

    # TF-IDF cosine JD vs each resume (resumes with no text get 0.0)
    corpus = [jd_text] + [p["text"] if p["text"].strip() else " " for p in parsed]
    vec = TfidfVectorizer(lowercase=True, stop_words="english",
                          ngram_range=(1, 2), max_features=5000)
    tfidf = vec.fit_transform(corpus)
    terms = vec.get_feature_names_out()
    jd_vec = tfidf[0]
    cosines = cosine_similarity(jd_vec, tfidf[1:])[0]

    results = []
    for p, cos in zip(parsed, cosines):
        matched = p["skills"] & jd_skills
        missing = jd_skills - p["skills"]
        other = p["skills"] - jd_skills
        skill_score = (len(matched) / len(jd_skills)) if jd_skills else 0.0
        score = 100.0 * (W_SKILL * skill_score + W_TFIDF * float(cos))

        # shared keyword overlap: top terms by min(tfidf_jd, tfidf_resume)
        row = tfidf[parsed.index(p) + 1]
        jd_arr = jd_vec.toarray()[0]
        r_arr = row.toarray()[0]
        shared = [(terms[i], min(jd_arr[i], r_arr[i]))
                  for i in range(len(terms)) if jd_arr[i] > 0 and r_arr[i] > 0]
        shared.sort(key=lambda t: t[1], reverse=True)
        top_terms = [t for t, _ in shared[:8]]

        if p["method"] == "none":
            note = ("No extractable text found in this PDF (likely a scanned "
                    "image). OCR is not included, so it scored 0 — check it manually.")
        elif not jd_skills:
            note = ("The job description contained no recognized skills from "
                    "skills.txt, so ranking leans fully on keyword similarity.")
        else:
            note = ""

        results.append({
            "name": p["name"],
            "score": round(score, 1),
            "skill_score": round(skill_score, 3),
            "tfidf_cosine": round(float(cos), 3),
            "explanation": {
                "matched_skills": sorted(matched),
                "missing_skills": sorted(missing),
                "other_skills": sorted(other),
                "jd_skills": sorted(jd_skills),
                "shared_keywords": top_terms,
                "n_shared_keywords": len(shared),
                "extract_method": p["method"],
                "note": note,
            },
        })

    results.sort(key=lambda r: (-r["score"], r["name"]))
    return results


if __name__ == "__main__":
    # tiny smoke test: rank the two-line fixtures below
    jd = "We need an accountant. Must know Tally Prime, GST filing and Excel."
    resumes = [
        {"name": "a.pdf",
         "pdf_bytes": b"%PDF-1.4 fake"},  # not a real pdf -> 'none', scores 0
    ]
    for r in rank_resumes(jd, resumes):
        print(r["name"], r["score"], r["explanation"]["note"][:60])
