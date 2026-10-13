# EVAL — recorded model outputs (2026-10-14)

Actual stdout of the tagger on the three sample photos. No mocking: every number below came out of mobilenetv2-12.onnx via onnxruntime.

## sample_photos/coffee_mug.jpg

```
{
  "photo": "sample_photos/coffee_mug.jpg",
  "model": "mobilenetv2-12.onnx",
  "top_prediction": {
    "label": "coffee mug",
    "confidence": 0.4915,
    "category": "kitchen",
    "category_match": "coffee mug"
  },
  "predictions": [
    {
      "label": "coffee mug",
      "confidence": 0.4915,
      "category": "kitchen",
      "category_match": "coffee mug"
    },
    {
      "label": "cup",
      "confidence": 0.3304,
      "category": "kitchen",
      "category_match": "cup"
    },
    {
      "label": "espresso",
      "confidence": 0.1722,
      "category": "appliances",
      "category_match": "espresso"
    },
    {
      "label": "pitcher",
      "confidence": 0.0016,
      "category": "other",
      "category_match": null
    },
    {
      "label": "consomme",
      "confidence": 0.0008,
      "category": "other",
      "category_match": null
    }
  ],
  "note": "labels are generic ImageNet classes; category mapping is heuristic"
}
```

## sample_photos/backpack.jpg

```
{
  "photo": "sample_photos/backpack.jpg",
  "model": "mobilenetv2-12.onnx",
  "top_prediction": {
    "label": "backpack",
    "confidence": 0.9825,
    "category": "fashion",
    "category_match": "backpack"
  },
  "predictions": [
    {
      "label": "backpack",
      "confidence": 0.9825,
      "category": "fashion",
      "category_match": "backpack"
    },
    {
      "label": "mailbag",
      "confidence": 0.0061,
      "category": "other",
      "category_match": null
    },
    {
      "label": "bearskin",
      "confidence": 0.002,
      "category": "other",
      "category_match": null
    },
    {
      "label": "sleeping bag",
      "confidence": 0.002,
      "category": "other",
      "category_match": null
    },
    {
      "label": "trench coat",
      "confidence": 0.0012,
      "category": "fashion",
      "category_match": "coat"
    }
  ],
  "note": "labels are generic ImageNet classes; category mapping is heuristic"
}
```

## sample_photos/sneakers.jpg

```
{
  "photo": "sample_photos/sneakers.jpg",
  "model": "mobilenetv2-12.onnx",
  "top_prediction": {
    "label": "Loafer",
    "confidence": 0.3537,
    "category": "fashion",
    "category_match": "loafer"
  },
  "predictions": [
    {
      "label": "Loafer",
      "confidence": 0.3537,
      "category": "fashion",
      "category_match": "loafer"
    },
    {
      "label": "sandal",
      "confidence": 0.3222,
      "category": "fashion",
      "category_match": "sandal"
    },
    {
      "label": "clog",
      "confidence": 0.3167,
      "category": "fashion",
      "category_match": "clog"
    },
    {
      "label": "shoe shop",
      "confidence": 0.0023,
      "category": "fashion",
      "category_match": "shoe"
    },
    {
      "label": "holster",
      "confidence": 0.0023,
      "category": "other",
      "category_match": null
    }
  ],
  "note": "labels are generic ImageNet classes; category mapping is heuristic"
}
```

