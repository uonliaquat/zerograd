#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__



#include "./multi_head_attention.h"
#include "./linear.h"
#include "./layer_norm.h"
#include "../models/gpt2/multi_layer_perceptron.h"


typedef struct TransformerLayerWorkspace{
    Tensor output;
    Tensor resid1_output;
    Tensor resid2_output;
} TransformerLayerWorkspace;

typedef struct TransformerLayerParams{
    LayerNormParams ln_1;
    MultiHeadAttentionLayerParams attn;
    LayerNormParams ln_2;
    MultiLayerPerceptronParams mlp;
} TransformerLayerParams;

typedef struct TransformerLayer{
    char name[128];
    size_t n_heads;
    LayerNorm ln_1;
    MultiHeadAttentionLayer attn;
    LayerNorm ln_2;
    MultiLayerPerceptron mlp;
    TransformerLayerWorkspace workspace;
} TransformerLayer;



TransformerLayer transformer_layer_init(TransformerLayerParams *params, const size_t n_heads, char *name);
void transformer_layer_free(TransformerLayer *layer);
void transformer_layer_forward(TransformerLayer *layer, Tensor *x);
void transformer_layer_write(TransformerLayer *layer, Tensor **tensors, size_t *tensors_len);
//void transformer_layer_write(TransformerLayer *transformer_layer, const char *filename);

#endif