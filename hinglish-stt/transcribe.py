#!/usr/bin/env python3
"""Hinglish-optimized speech-to-text CLI (faster-whisper, CPU, int8).

The key trick vs plain `whisper` CLI: we force language="hi" and pass an
`initial_prompt` made of Hindi-English code-switched sentences. Whisper's
decoder conditions on the prompt, which measurably steers it toward keeping
both Hindi and English spans instead of collapsing everything to one script.

Usage:
    python transcribe.py samples/clip1.wav
    python transcribe.py samples/clip1.wav --no-prompt --language auto
    python transcribe.py audio.mp3 --json
"""

import argparse
import json
import os
import sys

CACHED_TINY = os.path.expanduser(
    "~/workspace/pixalified-voice-agent/models/whisper-tiny"
)

HINGLISH_PROMPT = (
    "mera naam Rahul hai and I work in Mumbai. "
    "kal meeting kitne baje hai? I need to prepare the presentation. "
    "yeh phone ka price kya hai, does it have warranty?"
)


def fmt_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


def load_model(model_name: str, model_dir: str | None):
    from faster_whisper import WhisperModel

    if model_dir:
        path = model_dir
    elif model_name == "tiny" and os.path.isdir(CACHED_TINY):
        # Reuse the already-downloaded tiny weights; avoids a re-download.
        path = CACHED_TINY
    else:
        path = model_name  # faster-whisper will fetch from HF on first use
    return WhisperModel(path, device="cpu", compute_type="int8")


def transcribe_file(model, audio_path: str, language: str, prompt: str | None,
                    beam_size: int = 5):
    """Returns (full_text, segments). language='auto' -> auto-detect."""
    lang = None if language == "auto" else language
    segments, info = model.transcribe(
        audio_path,
        language=lang,
        initial_prompt=prompt,
        beam_size=beam_size,
        temperature=0.0,
    )
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs)
    return text, segs, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", help="Audio file (wav/mp3/m4a/...)")
    ap.add_argument("--model", default="tiny",
                    help="faster-whisper model name (default: tiny)")
    ap.add_argument("--model-dir", default=None,
                    help="Path to a local faster-whisper model dir")
    ap.add_argument("--language", default="hi",
                    help="'hi' for Hinglish-optimized, 'auto' for default decoding")
    ap.add_argument("--prompt", default=HINGLISH_PROMPT,
                    help="initial_prompt text (default: Hinglish code-switch examples)")
    ap.add_argument("--no-prompt", action="store_true",
                    help="Disable initial_prompt (default-decoding mode)")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = ap.parse_args()

    if not os.path.isfile(args.audio):
        print(f"error: audio file not found: {args.audio}", file=sys.stderr)
        return 1

    model = load_model(args.model, args.model_dir)
    prompt = None if args.no_prompt else args.prompt
    text, segs, info = transcribe_file(model, args.audio, args.language, prompt)

    if args.json:
        print(json.dumps({
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "text": text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segs
            ],
        }, ensure_ascii=False, indent=2))
    else:
        print(f"[detected language: {info.language} "
              f"(p={info.language_probability:.2f}), duration={info.duration:.1f}s]")
        print(f"\n{text}\n")
        for s in segs:
            print(f"[{fmt_ts(s.start)} --> {fmt_ts(s.end)}] {s.text.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
