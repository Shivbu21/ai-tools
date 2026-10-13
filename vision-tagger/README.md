# Vision Tagger — on-device product photo tagger

Point it at a product photo, get back ImageNet labels + a best-effort shop
category. Everything runs locally on CPU (onnxruntime); no network after setup.

## One-command run

```bash
cd ~/workspace/ai-tools/vision-tagger
python3 -m venv venv && venv/bin/pip install -r requirements.txt
python download_model.py     # one-time: fetches mobilenetv2-12.onnx (~14MB) + ImageNet labels
venv/bin/python tag.py sample_photos/coffee_mug.jpg
```

Output is JSON on stdout: top-5 labels, confidences, and the heuristic shop
category for each.

## How it works

1. **Preprocess** — EXIF-aware load, resize short side to 256, center-crop
   224×224, normalize with ImageNet mean/std, NCHW float32.
2. **Inference** — `mobilenetv2-12.onnx` (ONNX Model Zoo, validated build)
   via onnxruntime CPU → 1000-class logits → softmax.
3. **Top-5** — the 5 highest-confidence ImageNet labels.
4. **Category mapping** — `categories.json` matches label substrings to shop
   categories (longest keyword wins). **This is a dumb heuristic, not AI:**
   `"espresso maker" -> appliances`, `"backpack" -> fashion`, anything
   unmatched -> `"other"`.

## Real example outputs (recorded 2026-10-14, actual model runs)

**coffee_mug.jpg** — Vintage VICTOR coffee mug:
| label | confidence | category |
|---|---|---|
| coffee mug | 0.4915 | kitchen |
| cup | 0.3304 | kitchen |
| espresso | 0.1722 | appliances |
| pitcher | 0.0016 | other |
| consomme | 0.0008 | other |

**backpack.jpg** — school backpack:
| label | confidence | category |
|---|---|---|
| backpack | 0.9825 | fashion |
| mailbag | 0.0061 | other |
| bearskin | 0.0020 | other |
| sleeping bag | 0.0020 | other |
| trench coat | 0.0012 | fashion |

**sneakers.jpg** — 1890s Goodyear sneakers (dark museum exhibit photo):
| label | confidence | category |
|---|---|---|
| Loafer | 0.3537 | fashion |
| sandal | 0.3222 | fashion |
| clog | 0.3167 | fashion |
| shoe shop | 0.0023 | fashion |
| holster | 0.0023 | other |

Full JSON for all three is in `EVAL.md`.

## Test photo sources

All three are Creative Commons / public-domain photos from Wikimedia Commons,
downloaded via `Special:FilePath` on 2026-10-14 and kept in `sample_photos/`:

| file | subject | source page | author | license |
|---|---|---|---|---|
| `coffee_mug.jpg` | Vintage VICTOR coffee mug | [File:Vintage VICTOR Coffee Mug.jpg](https://commons.wikimedia.org/wiki/File:Vintage_VICTOR_Coffee_Mug.jpg) | Quercus acuta | CC (own work) |
| `backpack.jpg` | School backpack | [File:School bag backpack.jpg](https://commons.wikimedia.org/wiki/File:School_bag_backpack.jpg) | Gamingforfun365 | CC (own work) |
| `sneakers.jpg` | 1890s Goodyear sneakers, Bata Shoe Museum | [File:Goodyear sneakers, 1890s …](https://commons.wikimedia.org/wiki/File:Goodyear_sneakers,_1890s,_Goodyear_Rubber_Manufacturing_Company_-_Bata_Shoe_Museum_-_DSC00745-001.JPG) | Daderot | CC0 |

## Honest limitations

- **Generic ImageNet labels, not fine-tuned for Indian products.** It knows
  "coffee mug" and "backpack" but has no concept of kirana packets, mithai
  boxes, kurtis, or pressure cookers as categories.
- **It cannot read text.** Hinglish/English text on packaging is invisible to
  it — for that you need an OCR stage (e.g. EasyOCR) alongside the classifier.
- **Small or cluttered objects fail.** Center-crop 224×224 discards context;
  a small product in a busy shelf photo will be mislabeled.
- **The category map is a keyword hack.** Adding `"sneaker" -> fashion` to a
  JSON file is curation, not learning. Real categorization needs fine-tuning —
  see `finetune_stub.py` for the (documented, not run) recipe.

## Files

- `tag.py` — the CLI tagger
- `download_model.py` — one-time model/label downloader
- `categories.json` — heuristic label → shop-category map
- `finetune_stub.py` — documented, NOT RUN recipe for fine-tuning on Indian products
- `sample_photos/` — the three CC test photos
- `EVAL.md` — full recorded outputs
- `requirements.txt` — onnxruntime, pillow, numpy
