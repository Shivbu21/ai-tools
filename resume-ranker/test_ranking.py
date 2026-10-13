#!/usr/bin/env python3
"""
test_ranking.py — the one-command demo AND the correctness check.

Generates the sample resumes (if missing), ranks all 5 against both sample
JDs, prints the ranked tables with explanations, and ASSERTS that:
  - for the accountant JD, priya_sharma_accountant.pdf ranks #1
    and amit_kumar_electrician.pdf ranks last
  - for the delivery JD, suresh_yadav_driver.pdf ranks #1 with a >30-point
    margin over #2, and priya_sharma_accountant.pdf (pure mismatch) ranks last
    (neha_gupta_assistant.pdf honestly scores above her — her objective mentions
    delivery work, which the ranker correctly picks up)

Run:  python test_ranking.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sample_data"))
sys.path.insert(0, str(ROOT))

import generate_samples  # noqa: E402
from ranker import rank_resumes  # noqa: E402

SAMPLE = ROOT / "sample_data"


def load_pdfs():
    resumes = []
    for pdf in sorted(SAMPLE.glob("*.pdf")):
        resumes.append({"name": pdf.name, "pdf_bytes": pdf.read_bytes()})
    return resumes


def show(title, jd_file, resumes):
    jd = (SAMPLE / jd_file).read_text(encoding="utf-8")
    results = rank_resumes(jd, resumes)
    print("=" * 78)
    print(title, f"({jd_file})")
    print("=" * 78)
    for i, r in enumerate(results, 1):
        e = r["explanation"]
        bar = "#" * int(r["score"] / 5)
        print(f"\n{i}. {r['name']}  score {r['score']:.1f}  [{bar}]")
        print(f"   skill overlap {r['skill_score']:.2f} | JD-keyword cosine {r['tfidf_cosine']:.3f} "
              f"| extracted via {e['extract_method']}")
        print(f"   matched skills : {', '.join(e['matched_skills']) or '—'}")
        print(f"   missing skills : {', '.join(e['missing_skills']) or '—'}")
        print(f"   shared keywords ({e['n_shared_keywords']}): "
              f"{', '.join(e['shared_keywords']) or '—'}")
        if e["note"]:
            print(f"   note: {e['note']}")
    return results


def main():
    if not list(SAMPLE.glob("*.pdf")):
        print("generating sample resumes...")
        generate_samples.main()
    resumes = load_pdfs()
    assert len(resumes) == 5, f"expected 5 sample PDFs, found {len(resumes)}"

    acc = show("ACCOUNTANT ranking", "jd_accountant.txt", resumes)
    assert acc[0]["name"] == "priya_sharma_accountant.pdf", \
        f"expected accountant best-fit first, got {acc[0]['name']}"
    assert acc[-1]["name"] == "amit_kumar_electrician.pdf", \
        f"expected electrician last for accountant JD, got {acc[-1]['name']}"

    drv = show("DELIVERY EXECUTIVE ranking", "jd_driver.txt", resumes)
    assert drv[0]["name"] == "suresh_yadav_driver.pdf", \
        f"expected driver best-fit first, got {drv[0]['name']}"
    gap = drv[0]["score"] - drv[1]["score"]
    assert gap > 30, f"expected a clear margin for the best fit, got gap={gap:.1f}"
    assert drv[-1]["name"] == "priya_sharma_accountant.pdf", \
        f"expected pure mismatch last for driver JD, got {drv[-1]['name']}"

    print("\n" + "=" * 78)
    print("ALL ASSERTIONS PASSED — ranking discriminates correctly on both roles.")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
