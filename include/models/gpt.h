#ifndef __GPT_H__
#define __GPT_H__

#include<stdbool.h>

#include "./tensor.h"
#include "../layers/embedding.h"
// #include "../layers/dropout.h"
// #include "../layers/linear.h"
// #include "../layers/layer_norm.h"
// #include "../layers/transformer.h"
#include "../layers/transformer.h"



typedef struct GPTConfig {
    size_t vocab_size;      // Vocab Size
    size_t context_len;     // Context Length
    size_t embed_dim;       // Embedding Dimensions
    size_t n_heads;         // No of Attention heads
    size_t n_layers;        // No of layers
    float drop_rate;       // Dropout rate
    bool qkv_bias;          // Query-Key-Value bias
    size_t batch_size;
    DataType dtype;
} GPTConfig;



typedef struct GPTParams{
    EmbeddingLayerParams wpe;
    EmbeddingLayerParams wte;
    TransformerLayerParams h[12];
    LayerNormParams ln_f;
    LinearLayerParams out_proj;
} GPTParams;

typedef struct GPTWrokspace{
    Tensor indices;
    Tensor position_indices;
    Tensor embeddings[13];
    Tensor next_token_prob_dist;
} GPTWrokspace;


typedef struct GPTModel{
    GPTConfig config;
    GPTParams *params;
    EmbeddingLayer wte_layer;
    EmbeddingLayer wpe_layer;
    TransformerLayer h_layer[12];
    LayerNorm ln_f_layer;
    LinearLayer out_proj_layer;
    GPTWrokspace workspace;
    Tensor output;
} GPTModel;


GPTModel model_gpt_init(GPTParams *params, 
    const size_t vocab_size, 
    const size_t context_len, 
    const size_t embed_dim, 
    const size_t n_heads, 
    const size_t n_layers,
    const float drop_rate, 
    const bool qkv_bias, 
    const size_t batch_size,
    const DataType dtype
);

void model_gpt_free(GPTModel *model);
void model_gpt_forward(GPTModel *model, Tensor *input);
// void model_gpt_safetensors_write(const char *filename, GPTParameters *params);



//void model_gpt_write(const char *path);


#endif