#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__


#include "./self_attention.h"
#include "./linear.h"
#include "./layer_norm.h"




typedef struct MLPParams{
    LinearLayerParams c_fc;
    LinearLayerParams c_proj;
} MLPParams;

typedef struct MLPWorkspace{
    Tensor gelu;
} MLPWorkspace;

typedef struct MLP{
    MLPParams *params;
    LinearLayer c_fc;
    LinearLayer c_proj;
    MLPWorkspace workspace;
    Tensor output;
} MLP;

typedef struct TransformerLayerParams{
    LayerNormParams     ln_1;
    LinearLayerParams   c_attn;
    LinearLayerParams   c_proj;
    LayerNormParams     ln_2;
    MLPParams           mlp;
} TransformerLayerParams;

typedef struct TransformerLayerWorkspace{
    Tensor residual_output[2];
    Tensor *qkv;
    Tensor *queries_heads;
    Tensor *keys_heads;
    Tensor *values_heads;
    Tensor context_vecs;
    Tensor *attn_layer_outputs;
} TransformerLayerWorkspace;

typedef struct TransformerLayer{
    TransformerLayerParams *params;

    LayerNorm           ln_1_layer;
    LinearLayer         c_attn_layer;

    SelfAttentionLayer  attn_layer[12];
    LinearLayer         c_proj_layer;
    LayerNorm           ln_2_layer;
    MLP                 mlp_layer;

    TransformerLayerWorkspace workspace;
    Tensor output;

    bool masked;
    size_t n_heads;
} TransformerLayer;



TransformerLayer transformer_layer_init(TransformerLayerParams *params, const size_t context_len, const size_t emebd_dim, const size_t n_heads, const bool masked, const DataType dtype, char *name);
void transformer_layer_free(TransformerLayer *transformer_layer);
void transformer_layer_forward(TransformerLayer *transformer_layer, Tensor *x);
void transformer_layer_write(TransformerLayer *transforemr_layer, Tensor **tensors, size_t *tensors_len);
void transformer_layer_print(TransformerLayer *transformer_layer, const char *heading);
//void transformer_layer_write(TransformerLayer *transformer_layer, const char *filename);

#endif