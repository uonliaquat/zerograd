"""
Run GPT-2 on token ids given as arguments and print the next token id.

Usage:
    python gpt2.py 15496 11 995
    python gpt2.py 12 12 4324 0 345 12
"""

import sys
import torch
from transformers import GPT2LMHeadModel

ids = [int(a) for a in sys.argv[1:]]
if not ids:
    raise SystemExit("usage: python gpt2.py <id> <id> ...")

model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
input_ids = torch.tensor([ids], dtype=torch.long)

with torch.no_grad():
    logits = model(input_ids).logits        # (1, seq_len, vocab)

next_id = int(logits[0, -1].argmax())        # greedy: argmax of last position
print("input ids:", ids)
print("next token id:", next_id)