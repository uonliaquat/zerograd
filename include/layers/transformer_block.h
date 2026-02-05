// #ifndef __TRANSFORMER_BLOCK_H__
// #define __TRANSFORMER_BLOCK_H__

// #include <stdlib.h>
// #include <stdbool.h>

// #include "./transformer_layer.h"
// #include "./layer_norm.h"
// #include "./tensor.h"

// typedef struct TransformerBlockParams{
//     SelfAttentionLayerParams attn;
// } TransformerBlockParams;


// typedef struct TransformerBlock{
//     TransformerBlockParams *params;
//     TransformerLayer *transformer_layers;
//     LayerNorm layer_norm;
//     size_t n_layers;
//     bool decoder;
// } TransformerBlock;


// TransformerBlock transformer_block_init(TransformerBlockParams *params, size_t context_len, size_t embed_dim, size_t n_heads, size_t n_layers, size_t drop_rate, bool qkv_bias, bool decoder);
// // TransformerBlock transformer_block_init(GPTConfig *config, GPTParameters *params);
// void transformer_block_free(TransformerBlock *transformer_block);
// void transformer_block_forward(TransformerBlock *transformer_block, Tensor *input);
// //void transformer_block_print(TransformerBlock *transformer_block, const char *heading);
// //void transformer_block_write(TransformerBlock *transformer_block, const char *base_path);
// #endif