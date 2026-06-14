"""
Validate the C GPT-2 graph against HuggingFace, node by node.

Input: all 1024 tokens = 0 (must match what the C run used).

Pass rule for each node:  abs_diff <= ATOL  OR  rel_diff <= RTOL
  - abs catches small-magnitude tensors (where relative error is noisy)
  - rel catches large-magnitude tensors (where fp32 accumulation inflates abs)
A real bug fails BOTH; pure float accumulation passes on rel.

Read top to bottom; the first real MISMATCH is the bug.
"""

import torch
from transformers import GPT2Model
from safetensors.torch import load_file

C_PATH = "/Users/uonliaquat/workspace/zerograd/my_model.safetensors"
S, NDIM, NHEAD = 1024, 768, 12
ATOL = 1e-4      # absolute floor
RTOL = 1e-4      # relative floor (fp32 ~1e-6..1e-5, so 1e-4 is comfortably above noise)

c  = load_file(C_PATH)
m  = GPT2Model.from_pretrained("gpt2").eval()
sd = m.state_dict()

PASS, FAIL = 0, 0

def check(name, expected):
    global PASS, FAIL
    if name not in c:
        print(f"  {name:30s} MISSING"); FAIL += 1; return
    e = expected.detach().reshape(-1).float()
    g = c[name].reshape(-1).float()
    if e.numel() != g.numel():
        print(f"  {name:30s} SHAPE hf={tuple(expected.shape)} c={tuple(c[name].shape)}")
        FAIL += 1; return

    abs_diff = (e - g).abs()
    max_abs  = abs_diff.max().item()
    scale    = e.abs().max().item()
    max_rel  = (max_abs / scale) if scale > 0 else max_abs

    ok = (max_abs <= ATOL) or (max_rel <= RTOL)
    verdict = "ok" if ok else "MISMATCH"
    print(f"  {name:30s} abs={max_abs:.6f}  rel={max_rel:.2e}  {verdict}")
    if ok: PASS += 1
    else:  FAIL += 1

with torch.no_grad():
    ids = torch.zeros((1, S), dtype=torch.long)
    pos = torch.arange(S).unsqueeze(0)

    print("== embeddings ==")
    check("wte", sd["wte.weight"])
    check("wpe", sd["wpe.weight"])
    tok, pe = m.wte(ids), m.wpe(pos)
    x = tok + pe
    check("token.embed", tok)
    check("pos.embed",   pe)
    check("input.embed", x)

    for i in range(12):
        b, p = m.h[i], f"h.{i}."
        print(f"== block {i} ==")

        check(f"{p}ln1.weight", sd[f"{p}ln_1.weight"])
        check(f"{p}ln1.bias",   sd[f"{p}ln_1.bias"])
        ln1 = b.ln_1(x)
        check(f"{p}ln1.out", ln1)

        cw, cb = sd[f"{p}attn.c_attn.weight"], sd[f"{p}attn.c_attn.bias"]
        check(f"{p}q.weight", cw[:, 0*NDIM:1*NDIM]); check(f"{p}q.bias", cb[0*NDIM:1*NDIM])
        check(f"{p}k.weight", cw[:, 1*NDIM:2*NDIM]); check(f"{p}k.bias", cb[1*NDIM:2*NDIM])
        check(f"{p}v.weight", cw[:, 2*NDIM:3*NDIM]); check(f"{p}v.bias", cb[2*NDIM:3*NDIM])
        qkv = b.attn.c_attn(ln1)
        q, k, v = qkv.split(NDIM, dim=2)
        check(f"{p}q.proj", q)
        check(f"{p}k.proj", k)
        check(f"{p}v.proj", v)

        attn = b.attn(ln1)[0]
        check(f"{p}attn.proj.weight", sd[f"{p}attn.c_proj.weight"])
        check(f"{p}attn.proj.bias",   sd[f"{p}attn.c_proj.bias"])
        check(f"{p}attn.proj", attn)

        resid1 = x + attn
        check(f"{p}res.conn1", resid1)

        check(f"{p}ln2.weight", sd[f"{p}ln_2.weight"])
        check(f"{p}ln2.bias",   sd[f"{p}ln_2.bias"])
        ln2 = b.ln_2(resid1)
        check(f"{p}ln2.out", ln2)

        check(f"{p}mlp.up.weight", sd[f"{p}mlp.c_fc.weight"])
        check(f"{p}mlp.up.bias",   sd[f"{p}mlp.c_fc.bias"])
        up = b.mlp.c_fc(ln2)
        check(f"{p}mlp.up.proj", up)

        gelu = b.mlp.act(up)
        check(f"{p}mlp.up.proj.gelu.out", gelu)

        check(f"{p}mlp.down.weight", sd[f"{p}mlp.c_proj.weight"])
        check(f"{p}mlp.down.bias",   sd[f"{p}mlp.c_proj.bias"])
        down = b.mlp.c_proj(gelu)
        check(f"{p}mlp.down.proj", down)

        x = resid1 + down
        check(f"{p}out", x)

    print("== final ==")
    check("ln.weight", sd["ln_f.weight"])
    check("ln.bias",   sd["ln_f.bias"])
    lnf = m.ln_f(x)
    check("ln.out", lnf)
    check("lm.head", lnf @ m.wte.weight.t())

print(f"\n{PASS} ok, {FAIL} mismatched")
