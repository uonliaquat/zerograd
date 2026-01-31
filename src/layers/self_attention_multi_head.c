
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


#include "../../include/layers/self_attention.h"
#include "../../include/layers/linear.h"
#include "../../include/utils.h"


static inline void self_attention_layer_workspace_init(SelfAttentionLayer *self_attention_layer){
    self_attention_layer->workspace.queries_chnuks      = calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.keys_chnuks         = calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.values_chnuks       = calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.context_vecs        = calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.attention_scores    = (Tensor){0};
    self_attention_layer->workspace.attention_scores_scaled = (Tensor){0};
    self_attention_layer->workspace.attention_weights   = (Tensor){0};
    self_attention_layer->workspace.concat_heads        = (Tensor){0};
    self_attention_layer->workspace.keys_transposed     = (Tensor){0};
}

SelfAttentionLayer self_attention_layer_init(const size_t seq_len, const size_t embed_dim, const size_t n_heads, const bool bias, const bool requires_grad){
    SelfAttentionLayer self_attention_layer;
    self_attention_layer_workspace_init(&self_attention_layer);

    size_t head_dim = embed_dim / n_heads;
    self_attention_layer.W_query    = linear_layer_init(embed_dim, embed_dim, bias, requires_grad, DTYPE_DOUBLE);
    self_attention_layer.W_key      = linear_layer_init(embed_dim, embed_dim, bias, requires_grad, DTYPE_DOUBLE);
    self_attention_layer.W_value    = linear_layer_init(embed_dim, embed_dim, bias, requires_grad, DTYPE_DOUBLE);
    self_attention_layer.heads_proj = linear_layer_init(embed_dim, embed_dim, bias, requires_grad, DTYPE_DOUBLE);
    self_attention_layer.seq_len = seq_len;
    self_attention_layer.embed_dim = embed_dim;
    self_attention_layer.n_heads = n_heads;
    self_attention_layer.head_dim = head_dim;
    return self_attention_layer;
}

void self_attention_layer_free(SelfAttentionLayer *self_attention_layer){
    linear_layer_free(&self_attention_layer->W_key);
    linear_layer_free(&self_attention_layer->W_query);
    linear_layer_free(&self_attention_layer->W_value);
    linear_layer_free(&self_attention_layer->heads_proj);
}

// Tensor self_attention_layer_forward(const SelfAttentionLayer *self_attention_layer, const Tensor *x){
//     linear_layer_forward(&self_attention_layer->W_query, x);
//     linear_layer_forward(&self_attention_layer->W_key,   x);
//     linear_layer_forward(&self_attention_layer->W_value, x);

//     tensor_print(&queries, "queries");
//     tensor_print(&keys, "keys");
//     tensor_print(&values, "values");

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


void self_attention_layer_mult_head_forward(SelfAttentionLayer *self_attention_layer, const Tensor *x, bool masked){

    linear_layer_forward(&self_attention_layer->W_query, x);
    linear_layer_forward(&self_attention_layer->W_key,   x);
    linear_layer_forward(&self_attention_layer->W_value, x);

    tensor_chunk_(&self_attention_layer->W_query.output,   self_attention_layer->n_heads, 1, self_attention_layer->workspace.queries_chnuks);
    tensor_chunk_(&self_attention_layer->W_key.output,     self_attention_layer->n_heads, 1, self_attention_layer->workspace.keys_chnuks);
    tensor_chunk_(&self_attention_layer->W_value.output,   self_attention_layer->n_heads, 1, self_attention_layer->workspace.values_chnuks);

    // tensor_print(&self_attention_layer->W_query.output, "self_attention_layer->W_query.output");
    // tensor_print(&self_attention_layer->W_key.output, "self_attention_layer->W_key.output");
    // tensor_print(&self_attention_layer->W_value.output, "self_attention_layer->W_value.output");
    // tensor_print(&self_attention_layer->workspace.queries_chnuks[0],    "queries_chnuks");
    // tensor_print(&self_attention_layer->workspace.keys_chnuks[0],       "keys_chnuks");
    // tensor_print(&self_attention_layer->workspace.values_chnuks[0],     "values_chnuks");

    for(size_t head = 0; head < self_attention_layer->n_heads; head++){
            char heading[512] = "\0";
            snprintf(heading, 512, "HEAD %zu", head);
            print_centered_heading(heading);

            // K^T
            tensor_print(&self_attention_layer->workspace.keys_chnuks[head],"keys");
            tensor_transpose_(&self_attention_layer->workspace.keys_chnuks[head], &self_attention_layer->workspace.keys_transposed);
            tensor_print(&self_attention_layer->workspace.keys_transposed, "keys_transposed");

            // Q.K^T
            tensor_print(&self_attention_layer->workspace.queries_chnuks[head], "queries");
            tensor_dot_product_(&self_attention_layer->workspace.queries_chnuks[head], &self_attention_layer->workspace.keys_transposed, &self_attention_layer->workspace.attention_scores);
            tensor_print(&self_attention_layer->workspace.attention_scores, "attention_scores");
            
            // (Q.K^T)/sqrt(d)
            tensor_elementwise_scale_(&self_attention_layer->workspace.attention_scores, 1/sqrt(self_attention_layer->workspace.keys_chnuks[head].shape[1]), &self_attention_layer->workspace.attention_scores_scaled);
            tensor_print(&self_attention_layer->workspace.attention_scores_scaled, "attention_scores_scaled");
            
            // // if(masked == true){
            // //     tensor_tril_(&self_attention_layer->workspace.attention_scores, -INFINITY, &self_attention_layer->workspace.attention_scores_scaled);
            // //     tensor_print(&attention_scores_scaled, "attention_scores_scaled");
                    
            // // }
            // softmax((Q.K^T)/sqrt(d))
            tensor_softmax_(&self_attention_layer->workspace.attention_scores_scaled, 1, &self_attention_layer->workspace.attention_weights);
            tensor_print(&self_attention_layer->workspace.attention_weights, "attention_weights");
            
            //  softmax((Q.K^T)/sqrt(d)).V
            tensor_print(&self_attention_layer->workspace.values_chnuks[head], "values");
            tensor_dot_product_(&self_attention_layer->workspace.attention_weights, &self_attention_layer->workspace.values_chnuks[head],  &self_attention_layer->workspace.context_vecs[head]);
            tensor_print(&self_attention_layer->workspace.context_vecs[head], "context_vecs [head]");
            // printf("======================================================================================\n");
    }

    tensor_concat_(self_attention_layer->workspace.context_vecs, self_attention_layer->n_heads, 1, &self_attention_layer->workspace.concat_heads);
    tensor_print(&self_attention_layer->workspace.concat_heads, "concat_heads");
    linear_layer_forward(&self_attention_layer->heads_proj, &self_attention_layer->workspace.concat_heads);
    tensor_print(&self_attention_layer->heads_proj.output, "heads_proj (Output)");
}




void self_attention_layer_print(const SelfAttentionLayer *self_attention_layer, const char *heading){
    printf("\033[1;31m============================== %s ==============================\033[0m\n", heading);
    linear_layer_print(&self_attention_layer->W_query, "W_Query");
    linear_layer_print(&self_attention_layer->W_key, "W_key");
    linear_layer_print(&self_attention_layer->W_value, "W_value");
    linear_layer_print(&self_attention_layer->heads_proj, "heads_proj");
}


void self_attention_layer_write_fp(const SelfAttentionLayer *self_attention_layer, FILE *fptr){
    fprintf(fptr, "self_attention_multi_head\n");
    fprintf(fptr, "W_query\n");
    linear_layer_write_fp(&self_attention_layer->W_query, fptr);
    fprintf(fptr, "W_key\n");
    linear_layer_write_fp(&self_attention_layer->W_key, fptr);
    fprintf(fptr, "W_value\n");
    linear_layer_write_fp(&self_attention_layer->W_value, fptr);
    fprintf(fptr, "heads_proj\n");
    linear_layer_write_fp(&self_attention_layer->heads_proj, fptr);
}


void self_attention_layer_write(const SelfAttentionLayer *self_attention_layer, const char *base_path){
    char filename[512] = "\0";
    snprintf(filename, 512, "%s_w_query.csv", base_path);
    linear_layer_write(&self_attention_layer->W_query, filename);

    snprintf(filename, 512, "%s_w_key.csv", base_path);
    linear_layer_write(&self_attention_layer->W_key, filename);

    snprintf(filename, 512, "%s_w_value.csv", base_path);
    linear_layer_write(&self_attention_layer->W_value, filename);

    snprintf(filename, 512, "%s_heads_proj.csv", base_path);
    linear_layer_write(&self_attention_layer->heads_proj, filename);
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
