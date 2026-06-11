import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import sys

lib = ctypes.CDLL("./build/libkernels.dylib")

lib.kernel_multi_head_attention_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
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


num_tests = 100
passed = 0
for _ in range(num_tests):
    embed_dim = 786
    ctx_win = 1024
    n_heads = 12
    head_dim = embed_dim // n_heads

    q = np.random.randn(ctx_win, embed_dim).astype(np.float32)
    k = np.random.randn(ctx_win, embed_dim).astype(np.float32)
    v = np.random.randn(ctx_win, embed_dim).astype(np.float32)
    out = np.random.randn(ctx_win, embed_dim).astype(np.float32) 
    scratch = np.random.randn(ctx_win*embed_dim + ctx_win*ctx_win).astype(np.float32) 

    lib.kernel_multi_head_attention_cpu_f32_forward(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scratch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
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
