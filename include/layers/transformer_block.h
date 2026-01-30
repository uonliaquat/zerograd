#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#include <stdlib.h>
#include <stdbool.h>

typedef struct TransformerBlock{

} TransformerBlock;

TransformerBlock transformer_block_init(size_t n_heads, size_t n_layers, size_t drop_rate, bool qkv_bias);
#endif