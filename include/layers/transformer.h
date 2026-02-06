#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__


#include "./self_attention.h"
#include "./linear.h"
#include "./layer_norm.h"




typedef struct MLPParams{
    LinearLayerParams c_fc;
    LinearLayerParams c_proj;
} MLPParams;

typedef struct MLP{
    MLPParams *params;
    LinearLayer layer1;
    LinearLayer layer2;
    Tensor output;
} MLP;

typedef struct TransformerLayerParams{
    SelfAttentionLayerParams attn;
    LayerNormParams ln_[2];
    MLPParams mlp;
} TransformerLayerParams;

typedef struct TransformerLayerWorkspace{
    Tensor residual_output;
} TransformerLayerWorkspace;

typedef struct TransformerLayer{
    TransformerLayerParams *params;
    SelfAttentionLayer attn_layer;
    LayerNorm ln_layer[2];
    MLP mlp_layer;
    bool masked;
    TransformerLayerWorkspace workspace;
    Tensor output;
} TransformerLayer;



TransformerLayer transformer_layer_init(TransformerLayerParams *params, const size_t context_len, const size_t emebd_dim, const size_t n_heads, const bool masked, const DataType dtype);
void transformer_layer_free(TransformerLayer *transformer_layer);
void transformer_layer_forward(TransformerLayer *transformer_layer, Tensor *x);
void transformer_layer_print(TransformerLayer *transformer_layer, const char *heading);
//void transformer_layer_write(TransformerLayer *transformer_layer, const char *filename);

#endif