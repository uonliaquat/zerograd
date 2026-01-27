#ifndef __GPT_H__
#define __GPT_H__

#include<stdbool.h>

#include "../layers/embedding.h"


typedef struct GPTConfig {
    size_t vocab_size;      // Vocab Size
    size_t context_len;     // Context Length
    size_t embed_dim;       // Embedding Dimensions
    size_t n_heads;         // No of Attention heads
    size_t n_layers;        // No of layers
    double drop_rate;       // Dropout rate
    bool qkv_bias;          // Query-Key-Value bias
} GPTConfig;


typedef struct GPTModel{
    EmbeddingLayer token_embed_layer;
    EmbeddingLayer pos_embed_layer;
} GPTModel;


void model_gpt_init_config(size_t vocab_size, size_t context_len, size_t embed_len, size_t n_heads, size_t n_layers, double drop_rate, bool qkv_bias);
void model_gpt_init();
void model_gpt_forward(Tensor *input);
void model_gpt_write();

#endif