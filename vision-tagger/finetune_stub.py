"""
finetune_stub.py — a correct, documented OUTLINE for fine-tuning MobileNet
on Indian product photos. MARKED NOT RUN: this is a recipe, not executed code.

Why this is a stub: fine-tuning needs (a) a labeled dataset of Indian
products (thousands of photos across categories like kirana goods, mithai,
textiles, ayurvedic products), (b) GPU hours we don't have on this box,
and (c) the packaged-goods nuance (Hinglish text on packets, regional
packaging) that generic ImageNet labels miss entirely.

When you have data + GPU, the real steps are:

Dataset folder layout (ImageFolder convention):
    data/
        train/
            packaged_goods/   # *.jpg of e.g. Maggi, Parle-G, Amul packets
            appliances/       # mixer grinders, pressure cookers, ...
            fashion/          # kurtis, sneakers, ...
            kitchen/
            ...
        val/
            packaged_goods/
            ...

Training loop outline (torchvision):
    1. Load mobilenet_v2(weights=MOBILENET_V2_Weights.IMAGENET1K_V2).
    2. Replace the classifier head:
           model.classifier[1] = nn.Linear(1280, num_shop_categories)
       (only the head has new random weights; the backbone already knows
       edges/textures from ImageNet — that's "transfer learning").
    3. Freeze the backbone for a few epochs (train head only), then
       unfreeze the last blocks and fine-tune with a small LR (1e-4).
    4. Augmentations: RandomResizedCrop(224), RandomHorizontalFlip,
       ColorJitter — packaging photos vary in lighting/angle.
    5. Loss: CrossEntropyLoss. Optimizer: AdamW. Scheduler: cosine.
    6. Track val accuracy; export best checkpoint to ONNX:
           torch.onnx.export(model, dummy_input, "mobilenet_shop.onnx")
       ...and point tag.py at it with --model.

Honest expectation: with ~200+ clean photos per category, top-1 accuracy
on held-out shop photos in the 70-85% range is realistic; Hinglish text on
packets still won't be *read* — for that you'd add an OCR stage (EasyOCR)
before or beside the classifier.
"""

raise NotImplementedError(
    "This is a documented stub, not runnable code. "
    "See the module docstring for the real fine-tuning recipe."
)
