import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import sys

lib = ctypes.CDLL("./build/libkernels.dylib")

lib.kernel_linear_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_bool
]

lib.kernel_linear_cpu_f32_forward.restype = None

num_tests1 = 50
passed = 0
for _ in range(num_tests1):
    m = np.random.randint(1, 1000)
    n = np.random.randint(1, 1000)
    k = np.random.randint(1, 1000)

    m = 2
    n = 2
    k = 2

    input = np.random.randn(m, k).astype(np.float32)
    weights = np.random.randn(n, k).astype(np.float32)
    bias = np.random.randn(n).astype(np.float32)

    out = np.zeros(m*n, dtype=np.float32)

    lib.kernel_linear_cpu_f32_forward(
        input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        #None,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m,
        n,
        k,
        False
    )   
    try:
        y = F.linear(
            input=torch.tensor(input, dtype=torch.float32), 
            weight=torch.tensor(weights, dtype=torch.float32), 
            bias=torch.tensor(bias, dtype=torch.float32)
            #bias=None
        )
        np.testing.assert_allclose(out, y.flatten().numpy(), rtol=1e-6, atol=1e-6)
        passed +=1
    except AssertionError as e:
        print(e)
        # print("input")
        # print(input)
        # print("\nweight")
        # print(weights)
        # print("\n\nout")
        # print(out[:20])
        # print(y.flatten().numpy()[:20])
        # exit(1)
        pass


num_tests2 = 50
for _ in range(num_tests2):
    m = np.random.randint(1, 1000)
    n = np.random.randint(1, 1000)
    k = np.random.randint(1, 200)

    # m = 12
    # n = 10
    # k = 4

    input = np.random.randn(m, k).astype(np.float32)
    weights = np.random.randn(k, n).astype(np.float32)
    bias = np.random.randn(n).astype(np.float32)

    out = np.zeros(m*n, dtype=np.float32)

    lib.kernel_linear_cpu_f32_forward(
        input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        #None,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        m,
        n,
        k,
        True
    )   
    try:
        y = F.linear(
            input=torch.tensor(input, dtype=torch.float32), 
            weight=torch.tensor(weights.T, dtype=torch.float32), 
            bias=torch.tensor(bias, dtype=torch.float32)
            #bias=None
        )
        np.testing.assert_allclose(out, y.flatten().numpy(), rtol=1e-5, atol=1e-5)
        passed +=1
    except AssertionError as e:
        print(e)
        # print("input")
        # print(input)
        # print("\nweight")
        # print(weights.T)
        # print("\n\nout")
        # print(out[:20])
        # print(y.flatten().numpy()[:20])
        pass

num_tests = num_tests1 + num_tests2
if passed == num_tests:
    print(
        f"\033[92m[PASS]\033[0m kernel_linear_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(0)
else:
    print(
        f"\033[91m[FAIL]\033[0m kernel_linear_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(1)
