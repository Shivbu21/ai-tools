"""
Download the pretrained MobileNetV2 ONNX model (one-time setup step).

Model:  mobilenetv2-12.onnx from the ONNX Model Zoo
Source: https://github.com/onnx/models
        (validated/vision/classification/mobilenet — ~14MB, MIT-style open license)
Labels: imagenet_classes.txt from https://github.com/pytorch/hub (MIT)

Usage:
    python download_model.py

This downloads both files into the project directory. Run once before tag.py.
"""

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/"
    "validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
)
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

MODEL_PATH = os.path.join(HERE, "mobilenetv2-12.onnx")
LABELS_PATH = os.path.join(HERE, "imagenet_classes.txt")


def download(url, dest, expected_min_bytes):
    if os.path.exists(dest) and os.path.getsize(dest) >= expected_min_bytes:
        print(f"already present: {os.path.basename(dest)} ({os.path.getsize(dest):,} bytes)")
        return
    print(f"downloading {os.path.basename(dest)} ...")
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    size = os.path.getsize(tmp)
    if size < expected_min_bytes:
        raise RuntimeError(
            f"download looks wrong ({size} bytes) — got an error page instead? "
            f"URL: {url}"
        )
    os.replace(tmp, dest)
    print(f"saved {os.path.basename(dest)} ({size:,} bytes)")


def main():
    download(MODEL_URL, MODEL_PATH, expected_min_bytes=10_000_000)
    download(LABELS_URL, LABELS_PATH, expected_min_bytes=10_000)
    print("done. You can now run: python tag.py <photo.jpg>")


if __name__ == "__main__":
    sys.exit(main())
