#ifndef __GPT_H__
#define __GPT_H__

#include<stdbool.h>

#include "./tensor.h"
#include "../layers/embedding.h"
// #include "../layers/dropout.h"
// #include "../layers/linear.h"
// #include "../layers/layer_norm.h"
// #include "../layers/transformer.h"
#include "../layers/transformer_block.h"



typedef struct GPTConfig {
    size_t vocab_size;      // Vocab Size
    size_t context_len;     // Context Length
    size_t embed_dim;       // Embedding Dimensions
    size_t n_heads;         // No of Attention heads
    size_t n_layers;        // No of layers
    double drop_rate;       // Dropout rate
    bool qkv_bias;          // Query-Key-Value bias
} GPTConfig;



typedef struct GP2Wrokspace{
    Tensor position_indicies;
    Tensor input_embeddings;
} GP2Wrokspace;


typedef struct GPTModel{
    EmbeddingLayer token_embed_layer;
    EmbeddingLayer pos_embed_layer;
    TransformerBlock transformer_block;
    // DropoutLayer drop_embed_layer;
    // // Transformer Blocks
    // TransformerLayer *transformer_layers;
    // LayerNorm layer_norm;
    // LinearLayer out_head_layer;

    GP2Wrokspace workspace;
    Tensor output;
} GPTModel;



void model_gpt_config_init(size_t vocab_size, size_t context_len, size_t embed_len, size_t n_heads, size_t n_layers, double drop_rate, bool qkv_bias);
void model_gpt_init();
void model_gpt_forward(Tensor *input);
void model_gpt_write();
void model_gpt_config_print();

#endif