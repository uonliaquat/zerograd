
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../include/layers/self_attention.h"
#include "../../include/layers/linear.h"


SelfAttentionLayer self_attention_layer_init(const size_t seq_len, const size_t embed_dim, const size_t num_heads, const bool bias, const bool requires_grad, const DataType dtype){
    SelfAttentionLayer self_attention_layer;
    size_t head_dim = embed_dim / num_heads;

    self_attention_layer.W_query    = linear_layer_init(embed_dim, head_dim, bias, requires_grad, dtype);
    self_attention_layer.W_key      = linear_layer_init(embed_dim, head_dim, bias, requires_grad, dtype);
    self_attention_layer.W_value    = linear_layer_init(embed_dim, head_dim, bias, requires_grad, dtype);
    self_attention_layer.out_proj   = linear_layer_init(seq_len, embed_dim, bias, requires_grad, dtype);
    linear_layer_print(&self_attention_layer.W_query, "W_query");
    linear_layer_print(&self_attention_layer.W_key, "W_key");
    linear_layer_print(&self_attention_layer.W_value, "W_value");
    self_attention_layer.seq_len = seq_len;
    self_attention_layer.embed_dim = embed_dim;
    self_attention_layer.num_heads = num_heads;
    self_attention_layer.head_dim = head_dim;
    return self_attention_layer;
}

Tensor self_attention_layer_forward(const SelfAttentionLayer *self_attention_layer, const Tensor *x){
    Tensor queries = linear_layer_forward(&self_attention_layer->W_query, x);
    Tensor keys = linear_layer_forward(&self_attention_layer->W_key, x);
    Tensor values = linear_layer_forward(&self_attention_layer->W_value, x);

    // tensor_print(&queries, "queries");
    // tensor_print(&keys, "keys");
    // tensor_print(&values, "values");

    // attenion_scroes = Q.K^t
    Tensor keys_transposed = tensor_transpose(&keys);
    //tensor_print(&keys_transposed, "keys_transposed");
    Tensor attention_scores = tensor_dot_product(&queries, &keys_transposed);
    //tensor_print(&attention_scores, "attention_scores");
    Tensor attention_scores_scaled = tensor_scale(&attention_scores, 1/sqrt(keys.shape[1]));
    //tensor_print(&attention_scores_scaled, "attention_scores_scaled");
    Tensor attention_weights = tensor_softmax(&attention_scores_scaled, 1);
    //tensor_print(&attention_weights, "attention_weights");
    Tensor context_vec = tensor_dot_product(&attention_weights, &values);
    //tensor_print(context_vec)
    return context_vec;
}

Tensor self_attention_layer_mult_head_forward(const SelfAttentionLayer *self_attention_layer, const Tensor *x){
    Tensor queries = linear_layer_forward(&self_attention_layer->W_query, x);
    Tensor keys = linear_layer_forward(&self_attention_layer->W_key, x);
    Tensor values = linear_layer_forward(&self_attention_layer->W_value, x);

    //tensor_print(&queries, "queries");
    // tensor_print(&keys, "keys");
    // tensor_print(&values, "values");


    Tensor *queries_chnuks = tensor_chunk(&queries, self_attention_layer->num_heads, 1);
    Tensor *keys_chnuks = tensor_chunk(&keys, self_attention_layer->num_heads, 1);
    Tensor *values_chnuks = tensor_chunk(&values, self_attention_layer->num_heads, 1);

    Tensor *context_vecs = calloc(self_attention_layer->num_heads, sizeof(Tensor));
    for(size_t head = 0; head < self_attention_layer->num_heads; head++){
            Tensor keys_transposed = tensor_transpose(&keys_chnuks[head]);
            //tensor_print(&keys_transposed, "keys_transposed");
            Tensor attention_scores = tensor_dot_product(&queries_chnuks[head], &keys_transposed);
            //tensor_print(&attention_scores, "attention_scores");
            Tensor attention_scores_scaled = tensor_scale(&attention_scores, 1/sqrt(keys_chnuks[head].shape[1]));
            //tensor_print(&attention_scores_scaled, "attention_scores_scaled");
            Tensor attention_weights = tensor_softmax(&attention_scores_scaled, 1);
            //tensor_print(&attention_weights, "attention_weights");
            Tensor context_vec = tensor_dot_product(&attention_weights, &values);
            //tensor_print(context_vec)
            context_vecs[head] = context_vec;

            //tensor_print(&context_vecs[head], "head");
    }

    Tensor concat_heads = tensor_concat(context_vecs, self_attention_layer->num_heads, 1);
    tensor_print(&concat_heads, "concat_heads");
    Tensor projected_context_vecs = linear_layer_forward(&self_attention_layer->out_proj, &concat_heads);
    tensor_print(&projected_context_vecs, "projected_context_vecs");
    

    return projected_context_vecs;
}


// void self_attention_layer_print(const SelfAttentionLayer *self_attention_layer, const char *heading){
//     for(size_t i = 0; i < self_attention_layer->num_heads; i++){
//         printf("\033[36m==============================HEAD %d==============================\033[0m\n", heading);
//         printf("\033[36m==============================LINEAR LAYER %s (WEIGHTS)==============================\033[0m\n", heading);
//         linear_layer_print(&self_attention_layer->W_query[i], "W_Query");
//         linear_layer_print(&self_attention_layer->W_key[i], "W_key");
//         linear_layer_print(&self_attention_layer->W_value[i], "W_value");
//     }
// }


void self_attention_layer_write(const SelfAttentionLayer *self_attention_layer, const char *filename){
    FILE *fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Couldn't open file %s\n", filename);
        exit(1);
    }
    fprintf(fptr, "%s\n", "W_Query");
    linear_layer_write_fp(&self_attention_layer->W_query, fptr);
    fprintf(fptr, "%s\n", "W_Key");
    linear_layer_write_fp(&self_attention_layer->W_key, fptr);
    fprintf(fptr, "%s\n", "W_Value");
    linear_layer_write_fp(&self_attention_layer->W_value, fptr);
    fprintf(fptr, "\n");
}


// Tensor self_attention_simplified(Tensor *input_embeddings){

//     Tensor input_embeddings_transposed = tensor_transpose(input_embeddings);
//     tensor_print(&input_embeddings_transposed);
//     tensor_write(&input_embeddings_transposed, "./output/input_embeddings_transposed.csv");

//     Tensor attention_scores = tensor_dot_product(input_embeddings, &input_embeddings_transposed);
//     tensor_print(&attention_scores);

//     Tensor attention_weights = tensor_softmax(&attention_scores, 1);
//     tensor_print(&attention_weights);

//     Tensor context_vectors = tensor_dot_product(&attention_weights, input_embeddings);
//     tensor_print(&context_vectors);

//     return input_embeddings_transposed;
// }
