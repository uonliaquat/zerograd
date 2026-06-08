import ctypes
import numpy as np
import sys

lib = ctypes.CDLL("./build/libkernels.dylib")


lib.kernel_add_cpu_f32_forward.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_size_t
];

lib.kernel_add_cpu_f32_forward.restype = None

num_tests = 1000
passed = 0
for _ in range(num_tests):
    n = np.random.randint(1, 10000)

    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)

    lib.kernel_add_cpu_f32_forward(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n
    )
    try:
        np.testing.assert_allclose(out, a + b, rtol=1e-6, atol=1e-6)
        passed +=1
    except AssertionError:
        pass

if passed == num_tests:
    print(
        f"\033[92m[PASS]\033[0m kernel_add_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(0)
else:
    print(
        f"\033[91m[FAIL]\033[0m kernel_add_cpu_f32_forward "
        f"({passed}/{num_tests} tests passed)"
    )
    sys.exit(1)


