#ifndef __TRANSFORMER_BLOCK_H__
#define __TRANSFORMER_BLOCK_H__

#include <stdlib.h>
#include <stdbool.h>
#include "./transformer.h"
#include "./tensor.h"


typedef struct TransformerBlock{
    TransformerLayer *transformer_layers;
    size_t n_layers;
    bool decoder;
} TransformerBlock;

TransformerBlock transformer_block_init(size_t n_heads, size_t n_layers, size_t drop_rate, bool qkv_bias, bool decoder);
void transformer_block_free(TransformerBlock *transformer_block);
void transformer_block_forward(TransformerBlock *transformer_block, Tensor *input);
void transformer_block_write(TransformerBlock *transformer_block, const char *base_path);
#endif