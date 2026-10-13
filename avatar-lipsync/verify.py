#!/usr/bin/env python3
"""Verify a lipsync output mp4:
  1. ffprobe: has a video stream AND an audio stream, duration > 0
  2. OpenCV: the mouth region — located from the SFD face box detected in the
     input portrait (same detector Wav2Lip uses) — differs across sampled
     frames (lips move) and differs from the static input portrait.

Usage: venv/bin/python verify.py --out results/demo.mp4 --face assets/portrait.png
Exit 0 = all checks pass, 1 = something failed.
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Wav2Lip"))  # bundled face_detection
from face_detection.detection.sfd.sfd_detector import SFDDetector  # noqa: E402


def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", path],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def face_box(det, face_path: str):
    boxes = det.detect_from_image(face_path)
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("no face detected in input portrait")
    b = max(boxes, key=lambda r: r[4])  # highest score; [x1,y1,x2,y2,score]
    return tuple(int(v) for v in b[:4])


def mouth_region(frame: np.ndarray, box) -> np.ndarray:
    """Lower-center of the detected face box, where the mouth lives."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return frame[int(y1 + 0.55 * h):int(y2 - 0.02 * h),
                 int(x1 + 0.20 * w):int(x2 - 0.20 * w)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--face", required=True)
    args = ap.parse_args()

    ok = True
    info = probe(args.out)
    kinds = [s["codec_type"] for s in info["streams"]]
    dur = float(info["format"].get("duration", 0))
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    print(f"streams: {kinds}")
    print(f"video: {vstream['codec_name']} {vstream['width']}x{vstream['height']} "
          f"@ {eval(vstream['avg_frame_rate']):.1f}fps, duration {dur:.2f}s")
    if "video" not in kinds or "audio" not in kinds:
        print("FAIL: missing video or audio stream"); ok = False
    if dur <= 0:
        print("FAIL: zero duration"); ok = False

    det = SFDDetector(device="cpu")
    box = face_box(det, args.face)
    print(f"detected face box in portrait: {box}")

    cap = cv2.VideoCapture(args.out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in sorted({int(n * f) for f in (0.05, 0.3, 0.55, 0.8)}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, fr = cap.read()
        if ret:
            frames.append(mouth_region(fr, box).astype(np.float32))
    cap.release()

    face = mouth_region(cv2.imread(args.face), box).astype(np.float32)

    diffs = [float(np.mean(np.abs(frames[i] - frames[i + 1])))
             for i in range(len(frames) - 1)]
    vs_input = [float(np.mean(np.abs(f - face))) for f in frames]
    print(f"frames sampled: {len(frames)}")
    print(f"mouth-region mean abs diff between sampled frames: "
          f"{[round(d, 2) for d in diffs]}")
    print(f"mouth-region mean abs diff vs input portrait: "
          f"{[round(d, 2) for d in vs_input]}")

    if not diffs or np.mean(diffs) < 1.5:
        print("FAIL: frames do not differ in the mouth region "
              "(lips not moving?)")
        ok = False
    if not vs_input or np.mean(vs_input) < 1.5:
        print("FAIL: output frames identical to input portrait")
        ok = False

    print("VERIFY: PASS" if ok else "VERIFY: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
