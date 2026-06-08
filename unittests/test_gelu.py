import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import sys

lib = ctypes.CDLL("./build/libkernels.dylib")

lib.kernel_gelu_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t
]

lib.kernel_gelu_cpu_f32_forward.restype = None

num_tests = 1000
passed = 0
for _ in range(num_tests):
    n = np.random.randint(1, 1000)
    x = np.random.randn(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)

    lib.kernel_gelu_cpu_f32_forward(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n
    )   
    try:
        y = F.gelu(torch.tensor(x, dtype=torch.float32), approximate="tanh")
        np.testing.assert_allclose(out, y.flatten().numpy(), rtol=1e-6, atol=1e-6)
        passed +=1
    except AssertionError as e:
        #print(e)
        print(out[:10])
        print(y.flatten().numpy()[:10])
        pass

if passed == num_tests:
    print(
        f"\033[92m[PASS]\033[0m kernel_gelu_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(0)
else:
    print(
        f"\033[91m[FAIL]\033[0m kernel_gelu_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(1)
