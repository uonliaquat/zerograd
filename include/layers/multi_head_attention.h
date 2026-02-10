#ifndef __SELF_ATTENTION_H__
#define __SELF_ATTENTION_H__

#include <stdbool.h>
#include "../tensor.h"
#include "./linear.h"


typedef struct MultiHeadAttentionLayerWorkspace{
    Tensor qkv[3];
    Tensor *q_heads;
    Tensor *k_heads;
    Tensor *v_heads;
    Tensor *keys_transposed;
    Tensor *attn_scores;
    Tensor *attn_scores_scaled;
    Tensor *attn_scores_masked;
    Tensor *attn_weights;
    Tensor *ctx_vecs;
    Tensor conact_vecs;
    Tensor output;
} MultiHeadAttentionLayerWorkspace;



typedef struct MultiHeadAttentionLayerParams{
    Tensor bias;
    LinearLayerParams c_attn;
    LinearLayerParams c_proj;
} MultiHeadAttentionLayerParams;


typedef struct MultiHeadAttentionLayer{
    char name[128];
    size_t n_heads;
    Tensor bias;
    LinearLayer c_attn;
    LinearLayer c_proj;
    MultiHeadAttentionLayerWorkspace workspace;
} MultiHeadAttentionLayer;


MultiHeadAttentionLayer multi_head_attention_layer_init(MultiHeadAttentionLayerParams *params, const size_t n_heads, char *name);
void                    multi_head_attention_layer_free(const MultiHeadAttentionLayer *layer);
void                    multi_head_attention_layer_forward(MultiHeadAttentionLayer *layer, Tensor *x);
void                    multi_head_attention_layer_write(MultiHeadAttentionLayer *layer, Tensor **tensors, size_t *tensors_len);

#endif