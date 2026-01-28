
// #include <stddef.h>
// #include <stdio.h>
// #include <math.h>
// #include <stdlib.h>
// #include <string.h>

// #include "../../include/layers/self_attention.h"
// #include "../../include/layers/linear.h"


// SelfAttentionLayer self_attention_layer_init(const size_t seq_len, const size_t embed_dim, const size_t num_heads, const bool bias, const bool requires_grad){
//     SelfAttentionLayer self_attention_layer;
//     size_t head_dim = embed_dim / num_heads;

//     self_attention_layer.W_query = linear_layer_init(embed_dim, embed_dim, bias, requires_grad, DTYPE_DOUBLE);
//     self_attention_layer.W_key = linear_layer_init(embed_dim, embed_dim, bias, requires_grad, DTYPE_DOUBLE);
//     self_attention_layer.W_value = linear_layer_init(embed_dim, embed_dim, bias, requires_grad, DTYPE_DOUBLE);
//     linear_layer_print(&self_attention_layer.W_query, "W_query");
//     linear_layer_print(&self_attention_layer.W_key, "W_key");
//     linear_layer_print(&self_attention_layer.W_value, "W_value");
//     self_attention_layer.seq_len = seq_len;
//     self_attention_layer.embed_dim = embed_dim;
//     self_attention_layer.num_heads = num_heads;
//     self_attention_layer.head_dim = head_dim;
//     return self_attention_layer;
// }

// Tensor self_attention_layer_forward(const SelfAttentionLayer *self_attention_layer, const Tensor *x){
//     Tensor queries = linear_layer_forward(&self_attention_layer->W_query, x);
//     Tensor keys = linear_layer_forward(&self_attention_layer->W_key, x);
//     Tensor values = linear_layer_forward(&self_attention_layer->W_value, x);

//     // tensor_print(&queries, "queries");
//     // tensor_print(&keys, "keys");
//     // tensor_print(&values, "values");

//     // attenion_scroes = Q.K^t
//     Tensor keys_transposed = tensor_transpose(&keys);
//     //tensor_print(&keys_transposed, "keys_transposed");
//     Tensor attention_scores = tensor_dot_product(&queries, &keys_transposed);
//     //tensor_print(&attention_scores, "attention_scores");
//     Tensor attention_scores_scaled = tensor_elementwise_scale(&attention_scores, 1/sqrt(keys.shape[1]));
//     //tensor_print(&attention_scores_scaled, "attention_scores_scaled");
//     Tensor attention_weights = tensor_softmax(&attention_scores_scaled, 1);
//     //tensor_print(&attention_weights, "attention_weights");
//     Tensor context_vec = tensor_dot_product(&attention_weights, &values);
//     //tensor_print(context_vec)
//     return context_vec;
// }
