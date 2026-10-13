#!/usr/bin/env python3
"""True lip-sync talking head via Wav2Lip — runs on CPU, end to end.

Usage:
    venv/bin/python lipsync.py --face assets/portrait.png \
        --audio assets/voice.wav --out results/demo.mp4

This is a thin wrapper around Wav2Lip/inference.py (run with cwd=Wav2Lip so
its local `audio`, `models` and `face_detection` packages import). Batch
sizes are kept small for a ~2.5GB RAM budget; raise them on bigger machines.
Slow on CPU (expect several minutes for a few seconds of audio) — that is
normal, not a bug.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(HERE, "Wav2Lip")
PY = os.path.join(HERE, "venv", "bin", "python")
DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "wav2lip_gan.pth")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--face", required=True,
                    help="Portrait image (png/jpg) or video with a face")
    ap.add_argument("--audio", required=True, help="wav/mp3 audio file")
    ap.add_argument("--out", required=True, help="Output mp4 path")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--wav2lip-batch-size", type=int, default=8,
                    help="Wav2Lip inference batch (default 8 for low RAM)")
    ap.add_argument("--face-det-batch-size", type=int, default=4)
    ap.add_argument("--pads", nargs=4, type=int, default=[0, 10, 0, 0],
                    help="Face-box padding: top bottom left right")
    args = ap.parse_args()

    for label, path in (("face", args.face), ("audio", args.audio),
                        ("checkpoint", args.checkpoint)):
        if not os.path.isfile(path):
            print(f"error: {label} not found: {path}", file=sys.stderr)
            if label == "checkpoint":
                print("run: bash setup.sh", file=sys.stderr)
            return 1
    if not os.path.isfile(PY):
        print(f"error: venv python not found at {PY} — run: bash setup.sh",
              file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cmd = [
        PY, os.path.join(WAV2LIP_DIR, "inference.py"),
        "--checkpoint_path", os.path.abspath(args.checkpoint),
        "--face", os.path.abspath(args.face),
        "--audio", os.path.abspath(args.audio),
        "--outfile", os.path.abspath(args.out),
        "--wav2lip_batch_size", str(args.wav2lip_batch_size),
        "--face_det_batch_size", str(args.face_det_batch_size),
        "--pads", *[str(p) for p in args.pads],
    ]
    print("running Wav2Lip inference on CPU ...")
    t0 = time.time()
    subprocess.run(cmd, cwd=WAV2LIP_DIR, check=True)
    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
