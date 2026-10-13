# How a Transformer Learns to Write — Concepts Behind Tiny GPT

This is the write-up for the build in `~/workspace/tiny-gpt/`
(`model.py`, `train.py`, `sample.py`). It explains, from zero, what is
actually happening inside the ~800k numbers that learned to write
Shakespeare-style text. Every example below is computed, not imagined —
the numbers check out in plain NumPy.

The single idea of all language models, big and small:

> **Look at some text. Guess the next character. Measure how wrong you
> were. Adjust. Repeat millions of times.**

---

## 1. Tokens: text becomes numbers

A neural network cannot eat letters; it eats numbers. The first thing
`train.py` does is build an **alphabet** — the ~65 distinct characters in
Tiny Shakespeare — and assign each one an integer:

```
chars = sorted(set(text))          # ['\n', ' ', '!', ..., 'a', 'b', ...]
stoi  = {ch: i for i, ch in enumerate(chars)}
```

So `"To be"` becomes something like `[20, 61, 1, 54, 57]`. This step is
called **tokenization**. Real models use sub-word tokens ("token" ≈ word
piece, not character) with vocabularies of 50k–200k, but the idea is
identical: text → integers → the model only ever sees integers.

## 2. Embeddings: numbers become vectors with "meaning"

An integer like `54` carries no meaning — the number 54 is not "closer" to
55 in any linguistic sense. So each token id is replaced by a **learned
vector** of 128 numbers (this is `self.token_emb = nn.Embedding(vocab_size, 128)`
in `model.py`):

```
token id 54 ("b")  ->  [0.31, -1.02, 0.44, ..., 0.87]   (128 numbers)
```

These vectors start random and get nudged during training until tokens
used in similar contexts end up with similar vectors. That is literally
what "the model learns the meaning of words" means at the mechanical
level: positions of points in a 128-dimensional space, moved around by
gradient descent.

**Positional embeddings** (`self.pos_emb`) solve a second problem: the
model sees a *set* of vectors, but order matters ("dog bites man" ≠ "man
bites dog"). So position `i` gets its own learned 128-number vector,
added on top:

```
input to the blocks = token_vector[i] + position_vector[i]
```

## 3. Attention: deciding what matters (worked example)

This is the famous part. When the model is about to guess the character
after `"cat sat"`, it should pay more attention to `"cat"` than to some
unrelated earlier word. Attention is a mechanism that lets each position
**look back at all previous positions and take a weighted mix of them** —
and the weights are computed from the data itself, not fixed.

For every position, the model computes three vectors from its input `x`
(`self.qkv` is one linear layer that produces all three at once):

- **query (q)** — "what am I looking for?"
- **key (k)** — "what do I contain?" (computed for every position)
- **value (v)** — "what do I contribute if picked?" (every position)

The attention weight from position `i` to position `j` is:

```
score(i -> j) = (q_i · k_j) / sqrt(head_size)
weights       = softmax(scores)          # scores -> percentages
output_i      = sum_j weights(i->j) * v_j
```

Dot product = similarity: if the query "looks like" a key, the score is
high. Dividing by `sqrt(head_size)` keeps the scores from exploding as
vectors get longer (a scaling detail that matters in practice).

### Tiny worked example (verified in NumPy)

Two positions, vectors of size 2. Position 2 wants to predict what comes
after `"cat sat"`:

```
q (position 2) = [1, 0]
k1 = [1, 0]   ("cat")        v1 = [2, 0]
k2 = [0, 1]   ("sat")        v2 = [0, 3]
```

Scores (dot product / sqrt(2)):

```
score(2->1) = (1*1 + 0*0) / 1.4142 = 0.7071
score(2->2) = (1*0 + 0*1) / 1.4142 = 0.0000
```

Softmax turns scores into weights (percentages that sum to 1):

```
weights = [e^0.7071, e^0] / (e^0.7071 + e^0)
        = [2.028, 1.000] / 3.028
        = [0.6698, 0.3302]
```

Position 2 pays ~67% attention to position 1 and ~33% to itself. The
output is the weighted mix of the values:

```
output = 0.6698 * [2, 0] + 0.3302 * [0, 3] = [1.3395, 0.9907]
```

That is attention. In the real model this happens with 128-dim vectors,
4 parallel **heads** (each learning a different notion of "relevance" —
one might track grammar, another long-range names), at every position, in
every one of the 4 layers. The **causal mask** (`torch.tril`) zeroes out
scores to future positions so the model can't cheat by peeking at the
answer — position `i` may only mix positions `<= i`.

The **MLP** after each attention (`Linear → GELU → Linear`, expanding
128 → 512 → 128) is the model's "thinking" step: attention *routes*
information, the MLP *transforms* it. The `x + ...` residual connections
let information skip past each sub-layer, which is what keeps deep
networks trainable at all.

## 4. The training loop: one game, millions of rounds

`train.py` runs this loop 3000 times:

```python
xb, yb = get_batch("train")   # 32 random chunks of 128 chars + the true next chars
_, loss = model(xb, yb)        # forward: guess, measure wrongness
optimizer.zero_grad()
loss.backward()                # backward: assign blame to every number
optimizer.step()               # nudge every number toward less-wrong
```

Concretely:

- **Batch**: 32 random 128-character windows of Shakespeare. The input `x`
  is characters 0–127 of each window; the target `y` is characters 1–128 —
  so at *every* position the model is asked "what comes next?", giving
  32 × 128 = 4096 simultaneous next-character quizzes per step.
- **Loss** (cross-entropy): at one position, the model outputs a
  probability for each of the 65 characters. If it assigned probability
  `p` to the *correct* next character, the loss there is `-log(p)`.
  Example: `p = 0.25` → loss `1.386`. A confident-correct guess
  (`p = 0.99`) gives loss `0.01`; a confident-wrong guess gives a huge
  loss. Average over all positions and the batch. Lower = better guesses.
- **Backprop + AdamW**: `loss.backward()` computes, for each of the
  ~800k parameters, which direction to move it to reduce the loss (the
  chain rule, applied mechanically). AdamW moves each parameter a small
  step (`lr = 3e-4`) in that direction, with per-parameter adaptive step
  sizes and decoupled weight decay. 3000 steps × 4096 quizzes ≈ **12.3
  million** next-character predictions in one training run.

`estimate_loss()` every 500 steps checks the model on the **validation**
split (the last 10% of the text, never trained on) — the honest test of
whether it learned patterns or just memorized. If train loss falls but
val loss rises, that's overfitting; here both fall together.

## 5. Sampling: writing with weighted dice (temperature)

`sample.py` / `model.generate()`: start with a prompt (`"\n"` or `"To be
or not"`), then repeat 400–500 times:

1. Run the model on the recent context → 65 scores (**logits**) for the
   next character.
2. Divide by **temperature** `T`, apply softmax → probabilities.
3. Roll a weighted die (`torch.multinomial`) → pick one character.
4. Append it, repeat.

Temperature is the creativity knob. For logits `[2.0, 1.0, 0.1]`:

| T | probabilities | effect |
|---|---|---|
| 0.5 | [0.864, 0.117, 0.019] | sharp: almost always the top pick, repetitive but safe |
| 1.0 | [0.659, 0.242, 0.099] | the model's honest distribution |
| 2.0 | [0.502, 0.304, 0.194] | flat: more surprising, more gibberish |

`T → 0` is greedy decoding (always the argmax — deterministic, dull).
The training run's demo and `sample.py` use `T = 0.8`: mostly sensible,
occasionally surprising. That is all "creativity" is in an LLM —
controlled randomness over learned probabilities.

## 6. Scale: where this sits in the world

| | Tiny GPT (this build) | GPT-3 scale |
|---|---|---|
| parameters | ~800k | 175,000,000k |
| training tokens seen | ~12M | ~300,000M |
| context window | 128 chars | 128k+ tokens |
| training hardware | 1 laptop CPU, <1 hr | 1000s of GPUs, months |

Same architecture family (embeddings → attention blocks → next-token
loss → temperature sampling). The big ones add sub-word tokenizers,
dozens of layers, and engineering — not a different idea.

## 7. What this model can and cannot do (honest)

- **Can**: spell, punctuate, mimic Shakespearean cadence and vocabulary,
  continue a prompt in-style for a few sentences.
- **Cannot**: reason, remember facts, stay coherent over long passages,
  or know anything outside Tiny Shakespeare. With 128 characters of
  context and 800k parameters it has no room for world knowledge — every
  "fact" it emits is a stylistic accident. This is the expected,
  documented behavior of a model this size, not a bug.
