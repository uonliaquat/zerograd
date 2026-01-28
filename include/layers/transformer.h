#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#include "./self_attention.h"
#include "./linear.h"


typedef struct FeedForwardNetwork{
    LinearLayer linear_layer_input;
    LinearLayer linear_layer_output;
} FeedForwardNetwork;

typedef struct TransformerLayer{
    SelfAttentionLayer self_attention_layer;
    FeedForwardNetwork feed_forward_network;

} TransformerLayer;

TransformerLayer transformer_layer_init(size_t context_len, size_t emebd_dim, size_t n_heads, bool bias, bool requires_grad);
Tensor transformer_layer_forward(TransformerLayer *transformer_layer, Tensor *x, bool masked);


#endif