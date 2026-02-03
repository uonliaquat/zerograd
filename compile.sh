#!/bin/bash
gcc -g -Iinclude src/main.c src/safetensors.c src/tensor.c src/layers/embedding.c src/layers/linear.c src/layers/dropout.c src/layers/layer_norm.c src/layers/self_attention_multi_head.c src/layers/transformer.c src/layers/transformer_block.c src/models/gpt.c -lm -O0
#!/bin/bash
# gcc -g -Iinclude src/tensor.c src/model_loader.c src/models/gpt.c src/main.c -lm -O0
