#include "../../include/layers/transformer.h"



static inline FeedForwardNetwork feed_forward_network_init(size_t input_dim, size_t hidden_dim, size_t out_dim, bool bias, bool requires_grad){
    FeedForwardNetwork feed_forward_network;
    feed_forward_network.linear_layer_input = linear_layer_init(input_dim, hidden_dim, bias, requires_grad, DTYPE_DOUBLE);
    feed_forward_network.linear_layer_output = linear_layer_init(hidden_dim, out_dim, bias, requires_grad, DTYPE_DOUBLE);
    return feed_forward_network;
}

static inline Tensor feed_forward_network_forward(FeedForwardNetwork *feed_forward_network, Tensor *x){
    Tensor output_layer1 = linear_layer_forward(&feed_forward_network->linear_layer_input, x);
    Tensor output_layer2 = linear_layer_forward(&feed_forward_network->linear_layer_output, &output_layer1);
    return output_layer2;
}

TransformerLayer transformer_layer_init(size_t context_len, size_t emebd_dim, size_t n_heads, bool bias, bool requires_grad){
    TransformerLayer transformer_layer;
    transformer_layer.self_attention_layer = self_attention_layer_init(context_len, emebd_dim, n_heads, bias, requires_grad);
    transformer_layer.feed_forward_network = feed_forward_network_init(emebd_dim, emebd_dim*4, emebd_dim, bias, requires_grad);
    return transformer_layer;
}

Tensor transformer_layer_forward(TransformerLayer *transformer_layer, Tensor *x, bool masked){
    Tensor transformer_layer_output = self_attention_layer_mult_head_forward(&transformer_layer->self_attention_layer, x, masked);
    Tensor output = feed_forward_network_forward(&transformer_layer->feed_forward_network, &transformer_layer_output);
    return output;

}