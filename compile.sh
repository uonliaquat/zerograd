#!/bin/bash
gcc -Iinclude src/main.c src/tokenizer.c src/tensor.c src/dataset.c src/utils.c src/layers/linear.c src/layers/embedding.c
