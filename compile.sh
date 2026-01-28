#!/bin/bash
gcc -g -Iinclude src/main.c src/utils.c src/tensor.c src/layers/embedding.c src/layers/linear.c src/layers/dropout.c src/layers/layer_norm.c src/layers/self_attention_multi_head.c  src/layers/transformer.c src/models/gpt.c -lm -O2
