#!/usr/bin/env python3
"""Evaluate default vs Hinglish-optimized decoding on the sample clips.

- baseline:  language auto-detect, no initial_prompt  (default decoding)
- optimized: language="hi", initial_prompt with code-switched examples

Both hypotheses are normalized (Devanagari -> Latin transliteration,
lowercase, punctuation stripped) before WER, so script choice alone is not
penalized. Prints a before/after table and saves results/eval_results.json.

Usage:
    python eval.py
    python eval.py --model base --samples-dir samples
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from normalize import normalize_text, wer  # noqa: E402
from transcribe import HINGLISH_PROMPT, load_model, transcribe_file  # noqa: E402


def run() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tiny")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--samples-dir",
                    default=os.path.join(HERE, "samples"))
    ap.add_argument("--results-dir",
                    default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    with open(os.path.join(args.samples_dir, "transcripts.json"),
              encoding="utf-8") as f:
        transcripts = json.load(f)

    model = load_model(args.model, args.model_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    rows = []
    tot_base = tot_opt = 0.0
    for fname in sorted(transcripts):
        ref = transcripts[fname]
        audio = os.path.join(args.samples_dir, fname)

        base_text, _, base_info = transcribe_file(
            model, audio, "auto", None)
        opt_text, _, _ = transcribe_file(
            model, audio, "hi", HINGLISH_PROMPT)

        ref_n = normalize_text(ref)
        base_w = wer(ref_n, normalize_text(base_text))
        opt_w = wer(ref_n, normalize_text(opt_text))
        tot_base += base_w
        tot_opt += opt_w
        rows.append({
            "clip": fname,
            "detected_lang": base_info.language,
            "baseline_wer": round(base_w, 4),
            "optimized_wer": round(opt_w, 4),
            "reference": ref,
            "baseline_hyp": base_text,
            "optimized_hyp": opt_text,
        })

    n = len(rows)
    avg_base, avg_opt = tot_base / n, tot_opt / n
    with open(os.path.join(args.results_dir, "eval_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"rows": rows,
                   "avg_baseline_wer": avg_base,
                   "avg_optimized_wer": avg_opt},
                  f, ensure_ascii=False, indent=2)

    print(f"\n{'clip':<12}{'baseline WER':>14}{'optimized WER':>16}   "
          f"{'detected lang (baseline)'}")
    print("-" * 70)
    for r in rows:
        print(f"{r['clip']:<12}{r['baseline_wer']:>14.3f}"
              f"{r['optimized_wer']:>16.3f}    {r['detected_lang']}")
    print("-" * 70)
    print(f"{'AVERAGE':<12}{avg_base:>14.3f}{avg_opt:>16.3f}")
    delta = (avg_base - avg_opt)
    rel = (delta / avg_base * 100) if avg_base else 0.0
    print(f"\nAbsolute WER change: {delta:+.3f} "
          f"({rel:+.1f}% relative vs baseline)")
    print("Note: WER computed after Devanagari->Latin normalization "
          "(see normalize.py).")
    return 0


if __name__ == "__main__":
    sys.exit(run())
