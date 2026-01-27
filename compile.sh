#!/bin/bash
gcc -Iinclude src/main.c src/tensor.c src/utils.c src/layers/embedding.c src/layers/linear.c src/layers/self_attention_multi_head.c src/models/gpt.c -lm -O2
