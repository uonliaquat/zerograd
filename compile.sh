#!/bin/bash
#gcc -g -Iinclude src/main.c src/safetensors.c src/tensor.c src/layers/embedding.c src/layers/linear.c src/layers/dropout.c src/layers/layer_norm.c src/layers/self_attention_multi_head.c src/layers/transformer.c src/layers/transformer_block.c src/models/gpt.c -lm -O0
#!/bin/bash
gcc -Iinclude src/tokenizer.c src/main.c src/tensor.c src/safetensors.c src/layers/linear.c src/layers/embedding.c src/layers/transformer.c   src/layers/layer_norm.c  src/layers/multi_head_attention.c src/models/gpt2/multi_layer_perceptron.c src/models/gpt2/gpt.c -lm -O3
