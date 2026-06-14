"""
Validate the C GPT-2 graph against HuggingFace.

Structure:
  1. Load the C tensors into one dict:           c_nodes[name]  -> tensor
  2. Run HF and collect the same nodes into:     hf_nodes[name] -> tensor
  3. Compare the two dicts in a single loop.

Input: all 1024 tokens = 0 (must match what the C run used).

Pass rule per node:  abs_diff <= ATOL  OR  rel_diff <= RTOL
  - abs catches small-magnitude tensors (relative error is noisy there)
  - rel catches large-magnitude tensors (fp32 accumulation inflates abs)
  A real bug fails BOTH; pure float accumulation passes on rel.
"""

import torch
from transformers import GPT2Model
from safetensors.torch import load_file

# ─────────────────────────── config ────────────────────────────
C_PATH    = "/Users/uonliaquat/workspace/zerograd/my_model.safetensors"
SEQ_LEN   = 1024
EMBED_DIM = 768
N_LAYERS  = 12
ABS_TOL   = 1e-4
REL_TOL   = 1e-4


# ───────────────── dict 1: C-computed nodes ────────────────────
def build_c_nodes(path):
    """name -> tensor, straight from the safetensors your C engine wrote."""
    return dict(load_file(path).items())


# ───────────────── dict 2: HF reference nodes ──────────────────
def build_hf_nodes():
    """Run HF forward on all-zero input; collect the same node names the C graph uses."""
    model = GPT2Model.from_pretrained("gpt2").eval()
    p = model.state_dict()
    nodes = {}

    with torch.no_grad():
        token_ids = torch.zeros((1, SEQ_LEN), dtype=torch.long)
        positions = torch.arange(SEQ_LEN).unsqueeze(0)

        # embeddings
        nodes["wte"] = p["wte.weight"]
        nodes["wpe"] = p["wpe.weight"]
        token_embed  = model.wte(token_ids)
        pos_embed    = model.wpe(positions)
        hidden       = token_embed + pos_embed
        nodes["token.embed"] = token_embed
        nodes["pos.embed"]   = pos_embed
        nodes["input.embed"] = hidden

        for i in range(N_LAYERS):
            block = model.h[i]
            L = f"h.{i}."

            # LayerNorm 1
            nodes[f"{L}ln1.weight"] = p[f"{L}ln_1.weight"]
            nodes[f"{L}ln1.bias"]   = p[f"{L}ln_1.bias"]
            ln1 = block.ln_1(hidden)
            nodes[f"{L}ln1.out"] = ln1

            # Q/K/V weights — fused c_attn (768, 2304) split along output cols
            cw, cb = p[f"{L}attn.c_attn.weight"], p[f"{L}attn.c_attn.bias"]
            for tag, lo in (("q", 0), ("k", 1), ("v", 2)):
                cols = slice(lo * EMBED_DIM, (lo + 1) * EMBED_DIM)
                nodes[f"{L}{tag}.weight"] = cw[:, cols]
                nodes[f"{L}{tag}.bias"]   = cb[cols]

            # Q/K/V outputs — split the fused projection
            q_proj, k_proj, v_proj = block.attn.c_attn(ln1).split(EMBED_DIM, dim=2)
            nodes[f"{L}q.proj"] = q_proj
            nodes[f"{L}k.proj"] = k_proj
            nodes[f"{L}v.proj"] = v_proj

            # attention output (post c_proj)
            attn_out = block.attn(ln1)[0]
            nodes[f"{L}attn.proj.weight"] = p[f"{L}attn.c_proj.weight"]
            nodes[f"{L}attn.proj.bias"]   = p[f"{L}attn.c_proj.bias"]
            nodes[f"{L}attn.proj"] = attn_out

            # residual 1
            resid1 = hidden + attn_out
            nodes[f"{L}res.conn1"] = resid1

            # LayerNorm 2
            nodes[f"{L}ln2.weight"] = p[f"{L}ln_2.weight"]
            nodes[f"{L}ln2.bias"]   = p[f"{L}ln_2.bias"]
            ln2 = block.ln_2(resid1)
            nodes[f"{L}ln2.out"] = ln2

            # MLP up + GELU
            nodes[f"{L}mlp.up.weight"] = p[f"{L}mlp.c_fc.weight"]
            nodes[f"{L}mlp.up.bias"]   = p[f"{L}mlp.c_fc.bias"]
            mlp_up   = block.mlp.c_fc(ln2)
            mlp_gelu = block.mlp.act(mlp_up)
            nodes[f"{L}mlp.up.proj"]          = mlp_up
            nodes[f"{L}mlp.up.proj.gelu.out"] = mlp_gelu

            # MLP down
            nodes[f"{L}mlp.down.weight"] = p[f"{L}mlp.c_proj.weight"]
            nodes[f"{L}mlp.down.bias"]   = p[f"{L}mlp.c_proj.bias"]
            mlp_down = block.mlp.c_proj(mlp_gelu)
            nodes[f"{L}mlp.down.proj"] = mlp_down

            # residual 2 -> next block input
            hidden = resid1 + mlp_down
            nodes[f"{L}out"] = hidden

        # final LN + lm head
        nodes["ln.weight"] = p["ln_f.weight"]
        nodes["ln.bias"]   = p["ln_f.bias"]
        final_ln = model.ln_f(hidden)
        nodes["ln.out"]  = final_ln
        nodes["lm.head"] = final_ln @ model.wte.weight.t()

    return nodes


# ───────────────────────── comparison ──────────────────────────
def diff(ref, got):
    """Return (max_abs, max_rel, passed) for two tensors, or None if shapes differ."""
    r = ref.detach().reshape(-1).float()
    g = got.reshape(-1).float()
    if r.numel() != g.numel():
        return None
    max_abs = (r - g).abs().max().item()
    scale   = r.abs().max().item()
    max_rel = max_abs / scale if scale > 0 else max_abs
    passed  = (max_abs <= ABS_TOL) or (max_rel <= REL_TOL)
    return max_abs, max_rel, passed


def compare(hf_nodes, c_nodes):
    n_pass = n_fail = 0
    # iterate in HF (graph) order so the printout reads top-to-bottom
    for name, ref in hf_nodes.items():
        if name not in c_nodes:
            print(f"  {name:30s} MISSING in C file")
            n_fail += 1
            continue

        result = diff(ref, c_nodes[name])
        if result is None:
            print(f"  {name:30s} SHAPE  hf={tuple(ref.shape)} c={tuple(c_nodes[name].shape)}")
            n_fail += 1
            continue

        max_abs, max_rel, passed = result
        print(f"  {name:30s} abs={max_abs:.6f}  rel={max_rel:.2e}  {'ok' if passed else 'MISMATCH'}")
        n_pass += passed
        n_fail += (not passed)

    print(f"\n{n_pass} ok, {n_fail} mismatched")


# ─────────────────────────── main ──────────────────────────────
if __name__ == "__main__":
    c_nodes  = build_c_nodes(C_PATH)
    hf_nodes = build_hf_nodes()
    compare(hf_nodes, c_nodes)