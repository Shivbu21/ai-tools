"""Generate text from the trained model. Usage: python3 sample.py "To be or not" """

import sys
import torch

from model import TinyGPT

text = open("data/input.txt").read()
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

model = TinyGPT(len(chars))
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

prompt = sys.argv[1] if len(sys.argv) > 1 else "\n"
start = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long)
out = model.generate(start, max_new_tokens=500, temperature=0.8)
print("".join(itos[i] for i in out[0].tolist()))
