"""
Compare ONLY the parameter weights in your C safetensors against HF GPT-2.
Ignores all computed activations. Confirms the loader/split put the right
bytes in the right tensor.
"""

import torch
from transformers import GPT2Model
from safetensors.torch import load_file

C_PATH = "/Users/uonliaquat/workspace/zerograd/my_model.safetensors"
NDIM = 768
TOL = 1e-4

c = load_file(C_PATH)
sd = GPT2Model.from_pretrained("gpt2").eval().state_dict()

def cmp(name, hf_tensor):
    if name not in c:
        print(f"{name:28s} MISSING in C file")
        return
    h = hf_tensor.detach().reshape(-1).float()
    t = c[name].reshape(-1).float()
    if h.numel() != t.numel():
        print(f"{name:28s} SHAPE hf={tuple(hf_tensor.shape)} c={tuple(c[name].shape)}")
        return
    d = (h - t).abs().max().item()
    flag = "  <-- MISMATCH" if d > TOL else "  ok"
    print(f"{name:28s} maxdiff={d:.6f}{flag}")

# embeddings
cmp("wte", sd["wte.weight"])
cmp("wpe", sd["wpe.weight"])

for i in range(12):
    p = f"h.{i}."
    print(f"--- block {i} ---")

    cmp(f"{p}ln1.weight", sd[f"{p}ln_1.weight"])
    cmp(f"{p}ln1.bias",   sd[f"{p}ln_1.bias"])

    # split fused c_attn (768,2304) -> q/k/v (768,768) along output cols
    cw = sd[f"{p}attn.c_attn.weight"]   # (768, 2304) (in,out)
    cb = sd[f"{p}attn.c_attn.bias"]     # (2304,)
    cmp(f"{p}q.weight", cw[:, 0*NDIM:1*NDIM])
    cmp(f"{p}k.weight", cw[:, 1*NDIM:2*NDIM])
    cmp(f"{p}v.weight", cw[:, 2*NDIM:3*NDIM])
    cmp(f"{p}q.bias",   cb[0*NDIM:1*NDIM])
    cmp(f"{p}k.bias",   cb[1*NDIM:2*NDIM])
    cmp(f"{p}v.bias",   cb[2*NDIM:3*NDIM])

    cmp(f"{p}attn.proj.weight", sd[f"{p}attn.c_proj.weight"])
    cmp(f"{p}attn.proj.bias",   sd[f"{p}attn.c_proj.bias"])

    cmp(f"{p}ln2.weight", sd[f"{p}ln_2.weight"])
    cmp(f"{p}ln2.bias",   sd[f"{p}ln_2.bias"])

    cmp(f"{p}mlp.up.weight",   sd[f"{p}mlp.c_fc.weight"])
    cmp(f"{p}mlp.up.bias",     sd[f"{p}mlp.c_fc.bias"])
    cmp(f"{p}mlp.down.weight", sd[f"{p}mlp.c_proj.weight"])
    cmp(f"{p}mlp.down.bias",   sd[f"{p}mlp.c_proj.bias"])

print("--- final ---")
cmp("ln.weight", sd["ln_f.weight"])
cmp("ln.bias",   sd["ln_f.bias"])