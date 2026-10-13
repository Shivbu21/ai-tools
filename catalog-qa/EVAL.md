# EVAL — 5 real questions, run 2026-10-14 against the live stack
(ChromaDB with 76 products, llama-server qwen2.5-1.5b on :11435)

## Q1 — price lookup
**Q:** What is the price of Tata Salt?
**mode:** llm
**A:** The price of Tata Salt is Rs 28 per 1 kg.
**Cited:** Tata Salt Iodised — Rs 28 per 1 kg [Grocery]
**Verdict:** PASS — correct product, correct price.

## Q2 — cheapest in category
**Q:** Which atta is the cheapest?
**mode:** llm
**A:** Aashirvaad Whole Wheat Atta is the cheapest atta at Rs 265 per 5 kg.
**Cited:** Aashirvaad Whole Wheat Atta — Rs 265 per 5 kg [Grocery]
**Verdict:** PASS — the only atta in the catalog; correctly identified, no invention of alternatives.

## Q3 — availability
**Q:** Do you have basmati rice?
**mode:** llm
**A:** Yes, we have India Gate Rozzana Basmati Rice available. It's priced at Rs 145 per 1 kg.
**Cited:** India Gate Rozzana Basmati Rice — Rs 145 per 1 kg [Grocery]
**Verdict:** PASS — yes/no answered from the catalog, not from model memory.

## Q4 — out of catalog (hallucination check)
**Q:** Do you sell iPhone 16?
**mode:** not-found (retrieval distance above threshold → no LLM call)
**A:** I don't have that in the shop's catalog. I can only answer about items we stock.
**Verdict:** PASS — refused to answer instead of hallucinating. This is the
most important behaviour in the system and it held.

## Q5 — cheapest with ranking nuance
**Q:** cheapest soap
**mode:** llm
**A:** Lifebuoy Total Soap at Rs 32 per 125 g is the cheapest soap in the catalog.
**Cited:** Lux Soft Touch Soap — Rs 36/150 g; Lifebuoy Total Soap — Rs 32/125 g; Dove Soap Bar — Rs 48/100 g
**Verdict:** PASS — note the model correctly picked Lifebuoy (Rs 32) even
though Lux ranked first in retrieval. The cited rows let a human verify.

---

## Q6 — cross-product comparison (new, 2026-10-14 recovery run)
**Q:** Which coffee is cheaper, Nescafe or Bru?
**mode:** llm
**A:** "Nescafe Classic Instant Coffee is cheaper at Rs 175 per 50 g compared to Bru Instant Coffee at Rs 168 per 50 g."
**Cited:** Nescafe Classic Instant Coffee — Rs 175/50 g; Bru Instant Coffee — Rs 168/50 g
**Verdict:** FAIL — retrieval was correct (both coffees, right prices) but the
model misread the comparison: Bru (Rs 168) is the cheaper one. This is the
documented 1.5B limitation (README "Honest limitations"): same pack size made
the comparison trivial for a human, but the small model swapped the winner.
**Follow-up check:** rephrasing as a yes/no question — "Is Bru Instant Coffee
cheaper than Nescafe Classic?" — returned the correct answer ("Bru Instant
Coffee is cheaper than Nescafe Classic Instant Coffee"), same cited rows.
Takeaway: open-ended comparison questions are unreliable at this model size;
yes/no or single-fact questions hold up.

## Q7 — availability, different category (new, 2026-10-14 recovery run)
**Q:** Do you have shampoo?
**mode:** llm
**A:** Yes, we have Clinic Plus Shampoo available. It's priced at Rs 99 per 340 ml.
**Cited:** Clinic Plus Shampoo — Rs 99/340 ml; Head & Shoulders Anti-Dandruff — Rs 199/340 ml; Surf Excel Easy Wash — Rs 118/1 kg
**Verdict:** PASS — yes/no answered from the catalog with the correct product
and price.

## Q8 — out of catalog, new item (new, 2026-10-14 recovery run)
**Q:** Do you sell batteries?
**mode:** not-found (best distance 1.292 > 1.15 threshold → no LLM call)
**A:** I don't have that in the shop's catalog. I can only answer about items we stock.
**Verdict:** PASS — refused instead of hallucinating, same as Q4.
