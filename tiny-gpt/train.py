"""
Training script: this is where the model actually LEARNS.

The loop is dead simple and repeats thousands of times:
    1. Grab a random chunk of Shakespeare text
    2. Ask the model to guess each next character
    3. Measure how wrong it was (the "loss")
    4. Nudge every number inside the model a tiny bit toward being less wrong

Watch the loss number go DOWN — that's learning happening.
"""

import torch

from model import TinyGPT, count_params

# ---------------- settings ----------------
block_size = 128      # how many characters of context the model sees
batch_size = 32       # how many chunks per training step
n_embd = 128          # size of each character's vector
n_head = 4            # attention heads
n_layer = 4           # transformer blocks stacked
max_iters = 3000      # training steps
eval_interval = 500   # print progress every N steps
learning_rate = 3e-4
device = "cpu"
torch.manual_seed(42)

# ---------------- data: characters -> numbers ----------------
text = open("data/input.txt").read()
chars = sorted(set(text))              # our alphabet (~65 characters)
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}   # char -> number
itos = {i: ch for i, ch in enumerate(chars)}   # number -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))               # 90% train, 10% test
train_data, val_data = data[:n], data[n:]
print(f"dataset: {len(data):,} characters, alphabet size: {vocab_size}")


def get_batch(split):
    """Grab a random batch of (input chunk, next-char chunk) pairs."""
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])          # what the model sees
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])  # the correct answers
    return x, y


@torch.no_grad()
def estimate_loss(model):
    """Check the model on data it ISN'T training on (the honest test)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(20)
        for k in range(20):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------- build & train ----------------
model = TinyGPT(vocab_size, block_size, n_embd, n_head, n_layer)
print(f"model parameters: {count_params(model):,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(1, max_iters + 1):
    if it % eval_interval == 0 or it == 1:
        losses = estimate_loss(model)
        print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)   # forward: make guesses, measure wrongness
    optimizer.zero_grad()
    loss.backward()           # backward: figure out blame for each number
    optimizer.step()          # nudge every number toward less-wrong

torch.save(model.state_dict(), "tiny-gpt.pt")
print("saved to tiny-gpt.pt")

# ---------------- show off ----------------
model.eval()
start = torch.tensor([encode("\n")], dtype=torch.long)
out = model.generate(start, max_new_tokens=400, temperature=0.8)
print("\n--- the model writes Shakespeare-ish text: ---")
print(decode(out[0].tolist()))
