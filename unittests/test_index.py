import ctypes
import numpy as np

lib = ctypes.CDLL("./build/libkernels.dylib")

lib.kernel_index_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t
]

lib.kernel_index_cpu_f32_forward.restype = None

num_tests = 10
passed = 0
for _ in range(num_tests):
    vocab_size = 50257
    embed_dim = 786
    ctx_win = 1024
    n = np.random.randint(1, ctx_win)
    table = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    indices = np.random.randint(0, vocab_size, size=n, dtype=np.int32)
    out = np.zeros(n*embed_dim, dtype=np.float32)

    lib.kernel_index_cpu_f32_forward(
        table.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        embed_dim,
        n
    )
    try:
        np.testing.assert_allclose(out, table[indices].flatten(), rtol=1e-6)
        passed +=1
    except AssertionError:
        pass

if passed == num_tests:
    print(
        f"\033[92m[PASS]\033[0m kernel_index_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
else:
    print(
        f"\033[91m[FAIL]\033[0m kernel_index_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
