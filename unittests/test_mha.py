import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import math
import sys



lib = ctypes.CDLL("./build/libkernels.dylib")

FloatPtr = ctypes.POINTER(ctypes.c_float)

lib.kernel_multi_head_attention_cpu_f32_forward.argtypes = [
    ctypes.POINTER(FloatPtr),  # q
    ctypes.POINTER(FloatPtr),  # k
    ctypes.POINTER(FloatPtr),  # v
    ctypes.POINTER(FloatPtr),  # out
    ctypes.POINTER(FloatPtr),  # scratch
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t
]

lib.kernel_multi_head_attention_cpu_f32_forward.restype = None


def mha_reference(q, k, v):
    """
    q, k, v:
        list of n_heads arrays
        each shape = (ctx_win, head_dim)
    returns:
        list of n_heads outputs
        each shape = (ctx_win, head_dim)
    """
    outputs = []
    for h in range(len(q)):
        q_t = torch.tensor(q[h], dtype=torch.float32)
        k_t = torch.tensor(k[h], dtype=torch.float32)
        v_t = torch.tensor(v[h], dtype=torch.float32)
        y = F.scaled_dot_product_attention(
            q_t.unsqueeze(0),
            k_t.unsqueeze(0),
            v_t.unsqueeze(0),
            is_causal=True
        ).squeeze(0)
        outputs.append(y.numpy())
    return outputs

# def mha_reference(q, k, v):
#     print("\n\n\n===========================")
#     print("q")
#     print(q[0].flatten()[:10])
#     print("k")
#     print(k[0].flatten()[:10])
#     print("v")
#     print(v[0].flatten()[:10])
#     q_t = torch.tensor(q[0], dtype=torch.float32)
#     k_t = torch.tensor(k[0], dtype=torch.float32)
#     v_t = torch.tensor(v[0], dtype=torch.float32)

#     k_trans = k_t.T
#     print("\nK^T")
#     print(k_trans.flatten()[:10])

#     scores = q_t @ k_trans
#     print("\nAttention Scores")
#     print(scores.flatten()[:10])

#     scores = scores / math.sqrt(head_dim)
#     print("\nScaled")
#     print(scores.flatten()[:10])

#     mask = torch.triu(
#         torch.ones(ctx_win, ctx_win, dtype=torch.bool),
#         diagonal=1
#     )

#     scores = scores.masked_fill(mask, float("-inf"))
#     print("\nMasked")
#     print(scores.flatten()[:10])

#     probs = torch.softmax(scores, dim=-1)
#     print("\nSoftmax")
#     print(probs.flatten()[:10])

#     y = probs @ v_t
#     print("\nOutput")
#     print(y.flatten()[:10])


num_tests = 10
passed = 0

for _ in range(num_tests):

    embed_dim = 786
    ctx_win = 1024
    n_heads = 12
    head_dim = embed_dim // n_heads

    q = [
        np.random.randn(ctx_win, head_dim).astype(np.float32)
        for _ in range(n_heads)
    ]

    k = [
        np.random.randn(ctx_win, head_dim).astype(np.float32)
        for _ in range(n_heads)
    ]

    v = [
        np.random.randn(ctx_win, head_dim).astype(np.float32)
        for _ in range(n_heads)
    ]

    out = [
        np.zeros((ctx_win, head_dim), dtype=np.float32)
        for _ in range(n_heads)
    ]

    scratch = [
        np.zeros((ctx_win, head_dim), dtype=np.float32)
        for _ in range(n_heads)
    ]

    q_ptrs = (FloatPtr * n_heads)()
    k_ptrs = (FloatPtr * n_heads)()
    v_ptrs = (FloatPtr * n_heads)()
    out_ptrs = (FloatPtr * n_heads)()
    scratch_ptrs = (FloatPtr * n_heads)()

    for h in range(n_heads):
        q_ptrs[h] = q[h].ctypes.data_as(FloatPtr)
        k_ptrs[h] = k[h].ctypes.data_as(FloatPtr)
        v_ptrs[h] = v[h].ctypes.data_as(FloatPtr)
        out_ptrs[h] = out[h].ctypes.data_as(FloatPtr)
        scratch_ptrs[h] = scratch[h].ctypes.data_as(FloatPtr)

    lib.kernel_multi_head_attention_cpu_f32_forward(
        q_ptrs,
        k_ptrs,
        v_ptrs,
        out_ptrs,
        scratch_ptrs,
        ctx_win,
        embed_dim,
        n_heads,
        head_dim
    )

    try:
        expected = mha_reference(q, k, v)
        for h in range(n_heads):
            np.testing.assert_allclose(out[h], expected[h], rtol=1e-5, atol=1e-5)
        passed += 1

    except AssertionError as e:
        print(e)
        pass

if passed == num_tests:
    print(
        f"\033[92m[PASS]\033[0m kernel_multi_head_attention_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(0)
else:
    print(
        f"\033[91m[FAIL]\033[0m kernel_multi_head_attention_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(1)