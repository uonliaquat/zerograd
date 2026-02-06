#ifndef __SELF_ATTENTION_H__
#define __SELF_ATTENTION_H__

#include <stdbool.h>
#include "../tensor.h"
#include "./linear.h"


typedef struct SelfAttentionLayerWorkspace{
    Tensor *attention_scores;
    Tensor *attention_weights;
    Tensor *keys_transposed;
    Tensor *attention_scores_scaled;
    Tensor *queries_chnuks;
    Tensor *keys_chnuks;
    Tensor *values_chnuks;
    Tensor *context_vecs;
    Tensor *qkv;
    // Tensor queries;
    // Tensor keys;
    // Tensor values;
} SelfAttentionLayerWorkspace;

typedef struct SelfAttentionLayerParams{
    Tensor bias;
    LinearLayerParams c_attn;
    LinearLayerParams c_proj;
} SelfAttentionLayerParams;

typedef struct SelfAttentionLayer{
    SelfAttentionLayerParams *params;
    LinearLayer c_attn_layer;
    LinearLayer c_proj_layer;
    size_t context_len;
    size_t embed_dim;
    size_t n_heads;
    size_t head_dim;
    SelfAttentionLayerWorkspace workspace;
    Tensor output;
} SelfAttentionLayer;


SelfAttentionLayer  self_attention_layer_init(SelfAttentionLayerParams *params, const size_t context_len, const size_t embed_dim, const size_t n_heads, const DataType dtype);
void                self_attention_layer_params_free(SelfAttentionLayerParams *params);
void                self_attention_layer_free(const SelfAttentionLayer *self_attention_layer);
Tensor              self_attention_layer_forward(const SelfAttentionLayer *self_attention_layer, const Tensor *x);
void                self_attention_layer_multi_head_forward(SelfAttentionLayer *self_attention_layer, Tensor *x, bool masked);
//void                self_attention_layer_print(const SelfAttentionLayer *self_attention_layer, const char *heading);

// Tensor self_attention_simplified(Tensor *input_embeddings); 

#endif