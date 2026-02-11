#!/bin/bash
gcc -Iinclude \
    -DACCELERATE_NEW_LAPACK \
    -DACCELERATE_LAPACK_ILP64 \
    src/tokenizer.c \
    src/main.c \
    src/tensor.c \
    src/safetensors.c \
    src/layers/linear.c \
    src/layers/embedding.c \
    src/layers/transformer.c \
    src/layers/layer_norm.c \
    src/layers/multi_head_attention.c \
    src/models/gpt2/multi_layer_perceptron.c \
    src/models/gpt2/gpt.c \
    -lm \
    -O3 \
    -march=armv8.4-a \
    -mtune=native \
    -ffast-math \
    -funroll-loops \
    -ftree-vectorize \
    -framework Accelerate \
    -o gpt2
#!/bin/bash
# gcc -Iinclude src/tokenizer.c src/main.c src/tensor.c src/safetensors.c src/layers/linear.c src/layers/embedding.c src/layers/transformer.c   src/layers/layer_norm.c  src/layers/multi_head_attention.c src/models/gpt2/multi_layer_perceptron.c src/models/gpt2/gpt.c -lm -O3
