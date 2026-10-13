#!/usr/bin/env python3
"""Generate the Hinglish eval sample clips with Piper TTS.

Writes 16 kHz mono wav clips to samples/ and ground-truth transcripts to
samples/transcripts.json.

Voice: hi_IN-priyamvada-medium (Piper Hindi voice, downloaded via
`python -m piper.download_voices`). HONEST CAVEAT: this is synthetic speech
read by a TTS voice, not natural human code-switching. It exercises the
decoder's script/language handling, but real-world accuracy on natural speech
will differ. The fine-tune pipeline (prep_data.py) exists precisely to move
beyond synthetic samples with real Common Voice Hindi recordings.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(HERE, "samples")
VOICE_DIR = os.path.join(HERE, "models", "piper")
VOICE_NAME = "hi_IN-priyamvada-medium"
PIPER_BIN = os.path.expanduser(
    "~/workspace/pixalified-voice-agent/venv/bin/piper")

# 3 short, natural Hinglish utterances (Latin script, as commonly typed).
SENTENCES = [
    "mera naam Rahul hai and I work in Mumbai as a software engineer",
    "kal subah meeting hai at ten o'clock, please presentation ready rakhna",
    "yeh phone ka price kya hai and does it come with warranty",
]


def synth(text: str, out_wav: str, voice_onnx: str) -> None:
    tmp = out_wav + ".tmp.wav"
    subprocess.run(
        [PIPER_BIN, "--model", voice_onnx, "--output_file", tmp],
        input=text.encode("utf-8"), check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Resample to 16 kHz mono for a clean, uniform eval set.
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp,
         "-ar", "16000", "-ac", "1", out_wav], check=True)
    os.remove(tmp)


def main() -> int:
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(VOICE_DIR, exist_ok=True)
    voice_onnx = os.path.join(VOICE_DIR, VOICE_NAME + ".onnx")
    if not os.path.isfile(voice_onnx):
        print(f"downloading Piper voice {VOICE_NAME} ...")
        subprocess.run(
            [sys.executable, "-m", "piper.download_voices",
             "--download-dir", VOICE_DIR, VOICE_NAME], check=True)

    transcripts = {}
    for i, sent in enumerate(SENTENCES, start=1):
        fname = f"clip{i}.wav"
        out = os.path.join(SAMPLES_DIR, fname)
        print(f"synthesizing {fname}: {sent}")
        synth(sent, out, voice_onnx)
        transcripts[fname] = sent

    with open(os.path.join(SAMPLES_DIR, "transcripts.json"), "w",
              encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(transcripts)} clips + transcripts.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
