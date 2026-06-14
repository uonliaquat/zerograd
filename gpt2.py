# """
# Run GPT-2 on token ids given as arguments and print the next token id.

# Usage:
#     python gpt2.py 15496 11 995
#     python gpt2.py 12 12 4324 0 345 12
# """

# import sys
# import torch
# from transformers import GPT2LMHeadModel

# ids = [int(a) for a in sys.argv[1:]]
# if not ids:
#     raise SystemExit("usage: python gpt2.py <id> <id> ...")

# model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
# input_ids = torch.tensor([ids], dtype=torch.long)

# with torch.no_grad():
#     logits = model(input_ids).logits        # (1, seq_len, vocab)

# next_id = int(logits[0, -1].argmax())        # greedy: argmax of last position
# print("input ids:", ids)
# print("next token id:", next_id)


"""
Greedy GPT-2 generation from token ids given as arguments.

Usage:
    python gpt2.py 15496 11 995              # generate 20 tokens (default)
    python gpt2.py --n 50 15496 11 995       # generate 50 tokens

Greedy = at each step take argmax of the last position's logits,
append it, and feed the extended sequence back in.
"""

import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- parse args: optional "--n K", then the token ids ---
args = sys.argv[1:]
n_new = 20
if len(args) >= 2 and args[0] == "--n":
    n_new = int(args[1])
    args = args[2:]

ids = [int(a) for a in args]
if not ids:
    raise SystemExit("usage: python gpt2.py [--n K] <id> <id> ...")

model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
tok   = GPT2Tokenizer.from_pretrained("gpt2")

seq = list(ids)
with torch.no_grad():
    for _ in range(n_new):
        logits  = model(torch.tensor([seq], dtype=torch.long)).logits  # (1, len, vocab)
        next_id = int(logits[0, -1].argmax())                          # greedy
        seq.append(next_id)

print("prompt ids:", ids)
print("generated ids:", seq[len(ids):])
print("full ids:", seq)
print("text:", repr(tok.decode(seq)))