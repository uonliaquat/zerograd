from setuptools import setup, Extension

module = Extension(
    "my_library",
    sources=["src/bindings.c", "src/core_functions.c"],  # Include both C files
)

setup(
    name="my_library",
    version="1.0",
    description="A Python C extension with separate files",
    ext_modules=[module],
)
