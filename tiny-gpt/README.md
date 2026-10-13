# Tiny GPT — a real transformer, trained from scratch on CPU

A nanoGPT-style character-level transformer (**824,320 parameters**) that
learned to write Shakespeare-style text by playing one game ~12 million
times: *guess the next character*. This folder is the finished artifact;
the working/training directory is `~/workspace/tiny-gpt/` (left untouched).

**The model in this folder is fully trained.** `model.pt` is the real
checkpoint from the completed training run below — not a placeholder.

## Final training results (actual run, 2026-10-14, CPU)

| step | train loss | val loss |
|---|---|---|
| 1 | 4.3671 | 4.3680 |
| 500 | 2.3703 | 2.3947 |
| 1000 | 2.0461 | 2.0927 |
| 1500 | 1.8613 | 1.9716 |
| 2000 | 1.7399 | 1.8803 |
| 2500 | 1.6580 | 1.8314 |
| **3000** | **1.5858** | **1.7728** |

Config: 4 layers, 4 heads, 128-dim embeddings, 128-char context, batch 32,
AdamW lr=3e-4, seed 42. Dataset: 1,115,394 chars of Tiny Shakespeare,
65-char alphabet, 90/10 train/val split. ~12.3M next-char predictions.
Loss curve: `loss.png`. Val tracks train (no overfitting); the curve was
still falling at step 3000, so longer training would help a bit.

## Run it

```bash
pip install -r requirements.txt   # torch CPU
python sample.py "To be or not"   # generate 500 chars from a prompt
python train.py                   # retrain from scratch (~60-90 min CPU) -> tiny-gpt.pt
```

(`sample.py` here loads `model.pt`; the original `sample.py` in
`~/workspace/tiny-gpt/` loads `tiny-gpt.pt` — same weights, renamed.)

## Real sample outputs (verbatim, temperature 0.8)

Prompt `"To be or not"`:
```
To be or nothing in theirs wise hild,
This the fell mile to him that becallows
That of being badied post like met to the said.

Lord:
Citixford more her fright woold butward,
Not be an the sober, England
To a sover of that berine him shall bessighness miner.
```

Prompt `"ROMEO:"`:
```
ROMEO:
What you will such irtude is the purtages of he?
Then you, cannot That speak reptented,
Which him life is mest to mother?

KING HENRY VI:
Than cordome to the her rather, he is bown in the
Wilts spout a torting: hind who of rather lack kind
```

Full outputs in `samples.txt`. The flavor is right — character names,
line breaks, early-modern diction ("doth", "thy", "hath"-style words) —
mixed with invented words ("becallows", "bessighness"). That is exactly
what an 800k-parameter char model does: style without substance.

## Learn how it works

`CONCEPTS.md` explains the whole thing from scratch, tied to this code:
tokens → embeddings → attention (**with a worked numeric example, all
numbers verified**) → the training loop → temperature sampling → where
this sits vs. GPT-3 scale. Written for the build-as-you-learn goal.

## Files

- `model.pt` — trained weights (3.6MB), the real checkpoint
- `model.py` — the GPT: embeddings, causal attention, transformer blocks
- `train.py` — training loop (saves `tiny-gpt.pt` on a fresh run)
- `sample.py` — generate text from `model.pt`
- `data/input.txt` — Tiny Shakespeare (needed to rebuild the char vocab)
- `CONCEPTS.md` — from-scratch transformer explanation
- `loss.png` — training/validation loss curve
- `samples.txt` — verbatim generated samples

## Honest limitations

- **It cannot reason or know facts.** 800k params and 128 chars of context
  leave no room for world knowledge — anything that reads like a fact is a
  stylistic accident. Do not use it for anything truthful.
- **Character-level, tiny context.** Coherence collapses after a few
  sentences; spelling is approximate ("woold", "boody").
- **Trained only on Tiny Shakespeare.** Its entire universe is
  early-modern English drama. Prompt it about anything else and you get
  Shakespeare-flavored gibberish.
- **Training is CPU-slow.** ~60-90 min for 3000 steps here; scaling this
  approach up meaningfully needs a GPU.
