"""
tag.py — on-device product photo tagger.

Pipeline (everything runs locally, no network after setup):
    photo.jpg
      -> PIL: resize to 256, center-crop 224x224, RGB
      -> normalize (ImageNet mean/std)
      -> MobileNetV2 ONNX (onnxruntime, CPU) -> 1000 class scores
      -> top-5 ImageNet labels + softmax confidences
      -> categories.json heuristic -> best-effort shop category
      -> JSON printed to stdout

Usage:
    python tag.py <photo.jpg> [--topk 5] [--model mobilenetv2-12.onnx]

Example:
    python tag.py sample_photos/mug.jpg
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))

# Standard ImageNet preprocessing constants (PyTorch convention)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_TO = 256
CROP_TO = 224


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if len(labels) != 1000:
        raise ValueError(f"expected 1000 ImageNet labels, got {len(labels)}")
    return labels


def load_categories(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess(image_path):
    """Resize -> center crop -> RGB float32 tensor (1, 3, 224, 224), normalized."""
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    w, h = img.size
    # Resize so the SHORT side is 256 (keep aspect ratio)
    scale = RESIZE_TO / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    # Center crop 224x224
    w2, h2 = img.size
    left, top = (w2 - CROP_TO) // 2, (h2 - CROP_TO) // 2
    img = img.crop((left, top, left + CROP_TO, top + CROP_TO))
    # Normalize: [0,255] -> [0,1] -> (x - mean) / std, then CHW
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def map_to_category(label, categories):
    """
    Best-effort keyword mapping: ImageNet label -> simple shop category.

    This is a HEURISTIC, not intelligence — it matches substrings like
    "espresso maker" -> "appliances". Many labels map to "other" or to a
    rough bucket. Documented as such in the README.
    """
    label_lc = label.lower()
    # Longest matching keyword wins ("espresso maker" beats "espresso").
    for keyword in sorted(categories["keywords"], key=len, reverse=True):
        if keyword in label_lc:
            return categories["keywords"][keyword], keyword
    return categories["default"], None


def main():
    parser = argparse.ArgumentParser(description="Tag a product photo with MobileNetV2.")
    parser.add_argument("photo", help="path to an image file")
    parser.add_argument("--topk", type=int, default=5, help="number of top labels (default 5)")
    parser.add_argument("--model", default=os.path.join(HERE, "mobilenetv2-12.onnx"))
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}\nrun: python download_model.py")
    labels_path = os.path.join(HERE, "imagenet_classes.txt")
    if not os.path.exists(labels_path):
        sys.exit(f"labels not found: {labels_path}\nrun: python download_model.py")

    import onnxruntime as ort  # imported late so --help works without the dep

    labels = load_labels(labels_path)
    categories = load_categories(os.path.join(HERE, "categories.json"))

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    x = preprocess(args.photo)
    logits = sess.run(None, {input_name: x})[0][0]
    probs = softmax(logits)
    topk = np.argsort(probs)[::-1][: args.topk]

    predictions = []
    for idx in topk:
        label = labels[int(idx)]
        category, matched = map_to_category(label, categories)
        predictions.append(
            {
                "label": label,
                "confidence": round(float(probs[idx]), 4),
                "category": category,
                "category_match": matched,
            }
        )

    result = {
        "photo": args.photo,
        "model": os.path.basename(args.model),
        "top_prediction": predictions[0],
        "predictions": predictions,
        "note": "labels are generic ImageNet classes; category mapping is heuristic",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
