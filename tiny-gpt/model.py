"""
Our tiny GPT — a miniature version of the same idea behind ChatGPT.

The whole trick of a language model is one simple game:
    look at some text, guess the NEXT character, check if you were right.

Do that millions of times and the model slowly learns spelling,
grammar, and even a bit of style. That's literally all "training" is.

Architecture (don't worry, each piece is small):
    characters -> numbers -> vectors -> transformer blocks -> guess next char
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    The famous 'attention' mechanism. Plain-English version:

    When guessing the next character, the model looks BACK at all the
    previous characters and decides which ones matter most right now.
    E.g. after "The cat sat on the", it should pay attention to "cat"
    to guess that "mat" or "rug" comes next.

    "Causal" just means: you can only look at the past, never the future.
    (No cheating by peeking at the answer.)
    """

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        # One linear layer produces queries, keys AND values all at once
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        # The "causal mask": a triangle that hides future positions
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores = "how much should position i care about position j?"
        scores = (q @ k.transpose(-2, -1)) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # turn scores into percentages

        # Mix the "values" according to those percentages
        out = (weights @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    """A tiny feed-forward network: the model's 'thinking' step after attending."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # a smooth on/off switch
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block = attention + thinking, each with a shortcut
    (the '+ x' lets information skip past, which keeps training stable)."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embd=128, n_head=4, n_layer=4):
        super().__init__()
        self.block_size = block_size
        # Every character gets a learnable vector (its "meaning")
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # Every position gets a vector too (so the model knows ORDER)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        # Final layer: convert back to "score for each possible next character"
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size): scores for next char

        if targets is None:
            return logits, None
        # How wrong were we? (cross-entropy = "penalty for bad guesses")
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate text: guess one char, append it, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # only look at recent past
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)  # weighted dice roll
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


def count_params(model):
    return sum(p.numel() for p in model.parameters())
