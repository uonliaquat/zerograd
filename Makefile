CC = gcc

DEBUG_FLAGS = -g -O0
SHARED_FLAGS = -O0 -shared -fPIC

EXECUTABLE = main
LIBRARY = build/libkernels.dylib

SRCS = main.c $(shell find src -name '*.c')

.PHONY: build build_tests run run_tests clean

build:
	$(CC) $(DEBUG_FLAGS) $(SRCS) -o $(EXECUTABLE)

build_tests:
	mkdir -p build
	$(CC) $(SHARED_FLAGS) src/kernels/*.c src/prims/*.c -o $(LIBRARY)

run: build
	./$(EXECUTABLE)

run_tests: build_tests
	python unittests/run_tests.py

clean:
	rm -f $(EXECUTABLE)
	rm -rf build