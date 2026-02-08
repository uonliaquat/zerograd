#ifndef __SELF_ATTENTION_H__
#define __SELF_ATTENTION_H__

#include <stdbool.h>
#include "../tensor.h"
#include "./linear.h"


typedef struct SelfAttentionLayerWorkspace{
    Tensor attention_scores;
    Tensor attention_weights;
    Tensor keys_transposed;
    Tensor attention_scores_scaled;
    Tensor context_vecs;
} SelfAttentionLayerWorkspace;

typedef struct SelfAttentionLayerParams{
    Tensor bias;
    LinearLayerParams c_attn;
    LinearLayerParams c_proj;
} SelfAttentionLayerParams;

typedef struct SelfAttentionLayer{
    // SelfAttentionLayerParams *params;
    DataType dtype;
    char name[128];
    SelfAttentionLayerWorkspace workspace;
    Tensor output;
} SelfAttentionLayer;


SelfAttentionLayer  self_attention_layer_init(const DataType dtype, char *name);
// void                self_attention_layer_params_free(SelfAttentionLayerParams *params);
void                self_attention_layer_free(const SelfAttentionLayer *self_attention_layer);
Tensor              self_attention_layer_forward(const SelfAttentionLayer *self_attention_layer, const Tensor *x);
void                self_attention_layer_multi_head_forward(SelfAttentionLayer *self_attention_layer, Tensor *queries, Tensor *keys, Tensor *values, bool masked);
void                self_attention_layer_write(SelfAttentionLayer *self_attention_layer, Tensor **tensors, size_t *tensors_len);

// Tensor self_attention_simplified(Tensor *input_embeddings); 

#endif