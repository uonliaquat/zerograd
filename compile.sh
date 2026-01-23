#!/bin/bash
gcc -Iinclude src/main.c src/tokenizer.c src/tensor.c src/dataset.c src/dataloader.c src/utils.c src/layers/embedding.c src/layers/linear.c src/layers/dropout.c src/layers/self_attention_multi_head.c -lm -O2
