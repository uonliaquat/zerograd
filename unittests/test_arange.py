import ctypes
import numpy as np
import sys


lib = ctypes.CDLL("./build/libkernels.dylib")

lib.kernel_arange_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_size_t
]

lib.kernel_arange_cpu_f32_forward.restype = None

num_tests = 1000
passed = 0

for _ in range(num_tests):
    n = np.random.randint(1, 10000)
    out = np.zeros(n, dtype=np.int32)

    lib.kernel_arange_cpu_f32_forward(
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n
    )
    try:
        np.testing.assert_allclose(out, np.arange(n), rtol=1e-6, atol=1e-6)
        passed += 1
    except AssertionError:
        pass

if passed == num_tests:
    print(
        f"\033[92m[PASS]\033[0m kernel_arange_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(0)
else:
    print(
        f"\033[91m[FAIL]\033[0m kernel_arange_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    ) 
    sys.exit(1)

