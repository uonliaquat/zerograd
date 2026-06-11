import ctypes
import numpy as np
import torch
import math
import sys

lib = ctypes.CDLL("./build/libkernels.dylib")

lib.kernel_multi_head_attention_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # q
    ctypes.POINTER(ctypes.c_float),  # k
    ctypes.POINTER(ctypes.c_float),  # v
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.POINTER(ctypes.c_float),  # scratch
    ctypes.c_size_t,                 # ctx_win
    ctypes.c_size_t,                 # embed_dim
    ctypes.c_size_t,                 # n_heads
    ctypes.c_size_t,                 # head_dim
]
lib.kernel_multi_head_attention_cpu_f32_forward.restype = None


def mha_reference(q_t, k_t, v_t, head_dim, ctx_win):
    """Single-head causal attention. q/k/v: (ctx_win, head_dim) tensors."""
    scores = (q_t @ k_t.T) / math.sqrt(head_dim)
    mask = torch.triu(torch.ones(ctx_win, ctx_win, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    y = probs @ v_t
    return y.numpy()


def scratch_floats(ctx_win, head_dim):
    """Must match the kernel's scratch layout: k_t + qk_t + head_out."""
    return (ctx_win * head_dim) + (ctx_win * ctx_win) + (ctx_win * head_dim)


def run_case(embed_dim, ctx_win, n_heads, num_tests=10):
    head_dim = embed_dim // n_heads
    passed = 0

    for _ in range(num_tests):
        q = np.random.randn(ctx_win, embed_dim).astype(np.float32)
        k = np.random.randn(ctx_win, embed_dim).astype(np.float32)
        v = np.random.randn(ctx_win, embed_dim).astype(np.float32)
        out = np.random.randn(ctx_win, embed_dim).astype(np.float32)
        scratch = np.random.randn(scratch_floats(ctx_win, head_dim)).astype(np.float32)

        lib.kernel_multi_head_attention_cpu_f32_forward(
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctx_win, embed_dim, n_heads, head_dim,
        )

        try:
            for h in range(n_heads):
                cols = slice(h * head_dim, (h + 1) * head_dim)
                expected = mha_reference(
                    torch.tensor(q[:, cols]),
                    torch.tensor(k[:, cols]),
                    torch.tensor(v[:, cols]),
                    head_dim, ctx_win,
                )
                np.testing.assert_allclose(
                    out[:, cols], expected, rtol=1e-4, atol=1e-5,
                    err_msg=f"head {h} mismatch",
                )
            passed += 1
        except AssertionError as e:
            print(e)

    return passed, num_tests


# (embed_dim, ctx_win, n_heads)
cases = [
    (768, 2,  1),    # simplest: single head, minimal context
    (768, 16, 1),    # single head, real causal masking over many keys
    (10,  8,  2),    # tiny multi-head, exercises per-head offset + scatter
    (768, 1024, 12),   # real GPT-2 config
]

all_ok = True
for embed_dim, ctx_win, n_heads in cases:
    passed, total = run_case(embed_dim, ctx_win, n_heads)
    tag = f"embed={embed_dim} ctx={ctx_win} heads={n_heads}"
    if passed == total:
        print(f"\033[92m[PASS]\033[0m {tag}  ({passed}/{total})")
    else:
        print(f"\033[91m[FAIL]\033[0m {tag}  ({passed}/{total})")
        all_ok = False

print()
if all_ok:
    print("\033[92m[PASS]\033[0m kernel_multi_head_attention_cpu_f32_forward (all cases)")
    sys.exit(0)
else:
    print("\033[91m[FAIL]\033[0m kernel_multi_head_attention_cpu_f32_forward")
    sys.exit(1)