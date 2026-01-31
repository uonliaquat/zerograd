#include "../../include/layers/transformer.h"



static inline FeedForwardNetwork feed_forward_network_init(size_t input_dim, size_t hidden_dim, size_t out_dim, bool bias, bool requires_grad){
    FeedForwardNetwork feed_forward_network;
    feed_forward_network.linear_layer_input = linear_layer_init(input_dim, hidden_dim, bias, requires_grad, DTYPE_DOUBLE);
    feed_forward_network.linear_layer_output = linear_layer_init(hidden_dim, out_dim, bias, requires_grad, DTYPE_DOUBLE);
    return feed_forward_network;
}

static inline void feed_forward_network_free(FeedForwardNetwork *feed_forward_network){
    linear_layer_free(&feed_forward_network->linear_layer_input);
    linear_layer_free(&feed_forward_network->linear_layer_output);
}

// static inline Tensor feed_forward_network_forward(FeedForwardNetwork *feed_forward_network, Tensor *x){
//     Tensor output_layer1 = linear_layer_forward(&feed_forward_network->linear_layer_input, x);
//     Tensor output_layer2 = linear_layer_forward(&feed_forward_network->linear_layer_output, &output_layer1);
//     return output_layer2;
// }

TransformerLayer transformer_layer_init(size_t context_len, size_t emebd_dim, size_t n_heads, bool bias, bool requires_grad){
    TransformerLayer transformer_layer;
    transformer_layer.self_attention_layer = self_attention_layer_init(context_len, emebd_dim, n_heads, bias, requires_grad);
    //self_attention_layer_print(&transformer_layer.self_attention_layer, " transformer_layer.self_attention_layer");
    transformer_layer.feed_forward_network = feed_forward_network_init(emebd_dim, emebd_dim*4, emebd_dim, bias, requires_grad);
    return transformer_layer;
}

void transformer_layer_free(TransformerLayer *transformer_layer){
    feed_forward_network_free(&transformer_layer->feed_forward_network);
    self_attention_layer_free(&transformer_layer->self_attention_layer);
}

void transformer_layer_forward(TransformerLayer *transformer_layer, Tensor *x, bool masked){
    self_attention_layer_mult_head_forward(&transformer_layer->self_attention_layer, x, masked);
    // tensor_print(&transformer_layer_output, "transformer_layer_output");
    // Tensor output = feed_forward_network_forward(&transformer_layer->feed_forward_network, &transformer_layer_output);
    // tensor_print(&output, "feed_forward_network_output");
    // tensor_free(&transformer_layer_output);
}

void transformer_layer_write(TransformerLayer *transformer_write_fp, const char *base_path){
    char filename[512] = "\0";
    snprintf(filename, 512, "%s__%s", base_path, "self_attention_layer");
    self_attention_layer_write(&transformer_write_fp->self_attention_layer, filename);

    //memcpy(self_attention_layer_path, base_path, strlen(base_path));
    snprintf(filename, 512, "%s%s", base_path, "__feed_forward_network_input.csv");
    linear_layer_write(&transformer_write_fp->feed_forward_network.linear_layer_input, filename);

    snprintf(filename, 512, "%s%s", base_path, "__feed_forward_network_ouput.csv");
    linear_layer_write(&transformer_write_fp->feed_forward_network.linear_layer_output, filename);
}