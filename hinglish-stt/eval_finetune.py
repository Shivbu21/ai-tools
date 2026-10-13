#!/usr/bin/env python3
"""Evaluate a fine-tuned Whisper checkpoint on the dev set (WER).

Loads the model saved by finetune.py (a transformers Whisper checkpoint),
transcribes every utterance in data/dev.jsonl, and reports WER with the same
normalization used in eval.py so numbers are comparable.

Usage:
    python eval_finetune.py --model-dir whisper-hinglish
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from normalize import normalize_text, wer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="Checkpoint dir written by finetune.py")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    args = ap.parse_args()

    import torch
    import soundfile as sf
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_dir).to(device).eval()

    rows = []
    with open(os.path.join(args.data_dir, "dev.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = 0.0
    print(f"\n{'utterance':<24}{'WER':>8}")
    print("-" * 40)
    for r in rows:
        audio, sr = sf.read(r["audio"])
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        feats = processor(audio, sampling_rate=16000,
                          return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            pred_ids = model.generate(feats, language="hi",
                                      task="transcribe")
        hyp = processor.batch_decode(pred_ids,
                                     skip_special_tokens=True)[0]
        w = wer(normalize_text(r["text"]), normalize_text(hyp))
        total += w
        print(f"{os.path.basename(r['audio']):<24}{w:>8.3f}")
    print("-" * 40)
    print(f"{'AVERAGE':<24}{total / len(rows):>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
