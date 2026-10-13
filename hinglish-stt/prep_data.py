#!/usr/bin/env python3
"""Build a code-switched fine-tuning manifest for Whisper.

Combines two sources into train.jsonl / dev.jsonl (90/10 split):
  1. Local clips: samples/*.wav + samples/transcripts.json (always works).
  2. Common Voice Hindi (optional): streams a slice of
     mozilla-foundation/common_voice_17_0 ("hi") from the HuggingFace Hub.
     Needs `huggingface-cli login` (the dataset is gated). If the download
     fails, the script prints what to do and continues with local clips only.

Manifest line format: {"audio": "<16kHz mono wav path>",
                       "text": "<transcript>", "source": "..."}

Usage:
    python prep_data.py                       # local clips only
    python prep_data.py --cv-samples 2000     # + 2000 Common Voice Hindi clips
    python prep_data.py --out-dir data
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def save_wav_16k(audio_array, sample_rate, out_path):
    import numpy as np
    import soundfile as sf
    import librosa

    arr = np.asarray(audio_array, dtype=np.float32)
    if sample_rate != 16000:
        arr = librosa.resample(arr, orig_sr=sample_rate, target_sr=16000)
    if arr.ndim > 1:  # mono-ize
        arr = arr.mean(axis=0)
    sf.write(out_path, arr, 16000, subtype="PCM_16")


def fetch_common_voice(n_samples: int, audio_dir: str):
    """Stream n_samples of Common Voice Hindi; returns list of (wav, text)."""
    from datasets import load_dataset

    print(f"streaming {n_samples} Common Voice Hindi samples ...")
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "hi",
                      split="train", streaming=True, trust_remote_code=False)
    out = []
    for i, ex in enumerate(ds):
        if i >= n_samples:
            break
        wav_path = os.path.join(audio_dir, f"cv_hi_{i:05d}.wav")
        save_wav_16k(ex["audio"]["array"], ex["audio"]["sampling_rate"],
                     wav_path)
        out.append((wav_path, ex["sentence"].strip()))
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{n_samples}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir",
                    default=os.path.join(HERE, "samples"))
    ap.add_argument("--cv-samples", type=int, default=0,
                    help="Number of Common Voice Hindi clips to add (0 = skip)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    items = []  # (wav_path, text, source)

    # 1) local synthetic Hinglish clips
    tpath = os.path.join(args.samples_dir, "transcripts.json")
    if os.path.isfile(tpath):
        with open(tpath, encoding="utf-8") as f:
            transcripts = json.load(f)
        for fname, text in sorted(transcripts.items()):
            src = os.path.join(args.samples_dir, fname)
            if os.path.isfile(src):
                items.append((src, text, "synthetic_hinglish"))
        print(f"local clips: {len(transcripts)}")
    else:
        print("warning: samples/transcripts.json not found, "
              "run generate_samples.py first")

    # 2) Common Voice Hindi (real human speech)
    if args.cv_samples > 0:
        try:
            for wav, text in fetch_common_voice(args.cv_samples, audio_dir):
                items.append((wav, text, "common_voice_hi"))
        except Exception as e:  # noqa: BLE001 - gated dataset, auth, network...
            print("\n[Common Voice download failed — continuing with local "
                  "clips only]")
            print(f"reason: {type(e).__name__}: {e}")
            print("To enable it: `huggingface-cli login` (accept the dataset "
                  "terms at\n"
                  "https://huggingface.co/datasets/mozilla-foundation/"
                  "common_voice_17_0), then re-run with --cv-samples N.\n")

    if not items:
        print("error: no training data collected", file=sys.stderr)
        return 1

    random.Random(args.seed).shuffle(items)
    n_dev = max(1, int(0.1 * len(items)))
    dev, train = items[:n_dev], items[n_dev:]

    for name, split in (("train", train), ("dev", dev)):
        path = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for wav, text, source in split:
                f.write(json.dumps({"audio": os.path.abspath(wav),
                                    "text": text,
                                    "source": source},
                                   ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(split)} utterances)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
