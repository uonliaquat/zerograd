#ifndef __SELF_ATTENTION_H__
#define __SELF_ATTENTION_H__

#include <stdbool.h>
#include "../tensor.h"
#include "./linear.h"


// typedef struct SelfAttentionWorkspace{
//     // Tensor queries;
//     // Tensor keys;
//     // Tensor values;
// } SelfAttentionWorkspace;

typedef struct SelfAttentionLayer{
    LinearLayer W_query;
    LinearLayer W_key;
    LinearLayer W_value;
    LinearLayer heads_proj;
    size_t seq_len;
    size_t embed_dim;
    size_t n_heads;
    size_t head_dim;

    // SelfAttentionWorkspace self_attention_workspace;
    Tensor output;
} SelfAttentionLayer;

SelfAttentionLayer self_attention_layer_init(const size_t seq_len, const size_t embed_dim, const size_t num_heads, const bool bias, const bool requires_grad);
void    self_attention_layer_free(SelfAttentionLayer *self_attention_layer);
Tensor  self_attention_layer_forward(const SelfAttentionLayer *self_attention_layer, const Tensor *x);
void    self_attention_layer_mult_head_forward(SelfAttentionLayer *self_attention_layer, const Tensor *x, bool masked);
void    self_attention_layer_print(const SelfAttentionLayer *self_attention_layer, const char *heading);
void    self_attention_layer_write(const SelfAttentionLayer *self_attention_layer, const char *filename);
void    self_attention_layer_write_fp(const SelfAttentionLayer *self_attention_layer, FILE *fptr);

// Tensor self_attention_simplified(Tensor *input_embeddings); 

#endif