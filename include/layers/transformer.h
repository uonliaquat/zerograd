#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#include "./self_attention.h"
#include "./linear.h"


typedef struct FeedForwardNetwork{
    LinearLayer layer1;
    LinearLayer layer2;
} FeedForwardNetwork;

typedef struct TransformerLayer{
    SelfAttentionLayer self_attention_layer;
    FeedForwardNetwork feed_forward_network;
} TransformerLayer;


TransformerLayer transformer_layer_init(size_t context_len, size_t emebd_dim, size_t n_heads, bool bias, bool requires_grad);
void transformer_layer_free(TransformerLayer *transformer_layer);
void transformer_layer_forward(TransformerLayer *transformer_write_fp, Tensor *x, bool masked);
void transformer_layer_print(TransformerLayer *transformer_layer, const char *heading);
//void transformer_layer_write(TransformerLayer *transformer_layer, const char *filename);

#endif