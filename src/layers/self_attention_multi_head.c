
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>


#include "../../include/layers/self_attention.h"
#include "../../include/layers/linear.h"
#include "../../include/utils.h"



static inline void self_attention_layer_workspace_init(SelfAttentionLayer *self_attention_layer){
    self_attention_layer->workspace.qkv                 = calloc(3, sizeof(Tensor));
    self_attention_layer->workspace.queries_chnuks      = calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.keys_chnuks         = calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.values_chnuks       = calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.context_vecs        = calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.attention_scores    = calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.attention_scores_scaled =  calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.attention_weights   =  calloc(self_attention_layer->n_heads, sizeof(Tensor));
    self_attention_layer->workspace.concat_heads        = (Tensor){0};
    self_attention_layer->workspace.keys_transposed     = calloc(self_attention_layer->n_heads, sizeof(Tensor));


    for(size_t i = 0; i < 3; i++) {
        tensor_reset(&self_attention_layer->workspace.qkv[i]);
    }

    for(size_t i = 0; i < self_attention_layer->n_heads; i++){
        tensor_reset(&self_attention_layer->workspace.queries_chnuks[i]);
        tensor_reset(&self_attention_layer->workspace.keys_chnuks[i]);
        tensor_reset(&self_attention_layer->workspace.values_chnuks[i]);
        tensor_reset(&self_attention_layer->workspace.context_vecs[i]);
        tensor_reset(&self_attention_layer->workspace.attention_scores[i]);
        tensor_reset(&self_attention_layer->workspace.attention_scores_scaled[i]);
        tensor_reset(&self_attention_layer->workspace.attention_weights[i]);
        tensor_reset(&self_attention_layer->workspace.keys_transposed[i]);
    }
}

static inline void self_attention_layer_workspace_free(const SelfAttentionLayerWorkspace *workspace){
    tensor_free(workspace->queries_chnuks);
    tensor_free(workspace->keys_chnuks);
    tensor_free(workspace->values_chnuks);
    tensor_free(workspace->context_vecs);
    tensor_free(workspace->attention_scores);
    tensor_free(workspace->attention_scores_scaled);
    tensor_free(workspace->attention_weights);
    tensor_free(workspace->keys_transposed);
    tensor_free(&workspace->concat_heads);
}

SelfAttentionLayer self_attention_layer_init(SelfAttentionLayerParams *params, const size_t context_len, const size_t embed_dim, const size_t n_heads, const DataType dtype){
    SelfAttentionLayer self_attention_layer;
    self_attention_layer.params = params;

    size_t head_dim = embed_dim / n_heads;
    self_attention_layer.context_len = context_len;
    self_attention_layer.embed_dim = embed_dim;
    self_attention_layer.n_heads = n_heads;
    self_attention_layer.head_dim = head_dim;

    self_attention_layer_workspace_init(&self_attention_layer);
    self_attention_layer.c_attn_layer = linear_layer_init(&params->c_attn, dtype);
    self_attention_layer.c_proj_layer = linear_layer_init(&params->c_proj, dtype);
    return self_attention_layer;
}

void self_attention_layer_params_free(SelfAttentionLayerParams *params){
    tensor_free(&params->bias);
    linear_layer_params_free(&params->c_attn);
    linear_layer_params_free(&params->c_proj);
}


void self_attention_layer_free(const SelfAttentionLayer *self_attention_layer){
    self_attention_layer_workspace_free(&self_attention_layer->workspace);
    self_attention_layer_params_free(self_attention_layer->params);
}

// Tensor self_attention_layer_forward(const SelfAttentionLayer *self_attention_layer, const Tensor *x){
//     linear_layer_forward(&self_attention_layer->W_query, x);
//     linear_layer_forward(&self_attention_layer->W_key,   x);
//     linear_layer_forward(&self_attention_layer->W_value, x);

//     tensor_print(&queries);
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


void self_attention_layer_multi_head_forward(SelfAttentionLayer *self_attention_layer, Tensor *x, bool masked){
    // x             ==> (1, 4, 768), 
    // c_attn_layer  ==> (768, 2304)
    // output        ==> (1, 4, 2304)
    //tensor_print(&self_attention_layer->params->c_attn.weight, "self_attention_layer->params->c_attn.weight");
    linear_layer_forward(&self_attention_layer->c_attn_layer, x);
    tensor_print(&self_attention_layer->c_attn_layer.workspace.output, "c_attn_layer.workspace.output");
    tensor_chunk_(
        &self_attention_layer->c_attn_layer.workspace.output,   
        3, 1, self_attention_layer->workspace.qkv
    );
    // tensor_print(&self_attention_layer->workspace.qkv[0], "query");
    // tensor_print(&self_attention_layer->workspace.qkv[1], "key");
    // tensor_print(&self_attention_layer->workspace.qkv[2], "value");

    tensor_chunk_(&self_attention_layer->workspace.qkv[0],   self_attention_layer->n_heads, 1, self_attention_layer->workspace.queries_chnuks);
    tensor_chunk_(&self_attention_layer->workspace.qkv[1],   self_attention_layer->n_heads, 1, self_attention_layer->workspace.keys_chnuks);
    tensor_chunk_(&self_attention_layer->workspace.qkv[2],   self_attention_layer->n_heads, 1, self_attention_layer->workspace.values_chnuks);

    // // tensor_print(&self_attention_layer->W_query.output, "self_attention_layer->W_query.output");
    // // tensor_print(&self_attention_layer->W_key.output, "self_attention_layer->W_key.output");
    // // tensor_print(&self_attention_layer->W_value.output, "self_attention_layer->W_value.output");
    tensor_print(&self_attention_layer->workspace.queries_chnuks[0],    "queries_chnuks");
    tensor_print(&self_attention_layer->workspace.keys_chnuks[0],       "keys_chnuks");
    tensor_print(&self_attention_layer->workspace.values_chnuks[0],     "values_chnuks");

    for(size_t head = 0; head < self_attention_layer->n_heads; head++){
            char heading[512] = "\0";
            snprintf(heading, 512, "HEAD %zu", head);
            print_centered_heading(heading);

            // // K^T
            tensor_transpose_(&self_attention_layer->workspace.keys_chnuks[head], &self_attention_layer->workspace.keys_transposed[head]);

            // // Q.K^T
            tensor_dot_product_(
                &self_attention_layer->workspace.queries_chnuks[head], 
                &self_attention_layer->workspace.keys_transposed[head], 
                &self_attention_layer->workspace.attention_scores[head]
            );

            // // (Q.K^T)/sqrt(d)
            tensor_elementwise_scale_(
                &self_attention_layer->workspace.attention_scores[head], 
                1/sqrt(self_attention_layer->workspace.keys_chnuks[head].shape[self_attention_layer->workspace.keys_chnuks[head].ndim-1]), 
                &self_attention_layer->workspace.attention_scores_scaled[head]
            );

            // // // // if(masked == true){
            // // // //     tensor_tril_(&self_attention_layer->workspace.attention_scores, -INFINITY, &self_attention_layer->workspace.attention_scores_scaled);
            // // // //     tensor_print(&attention_scores_scaled, "attention_scores_scaled");
                    
            // // // // }
            // // softmax((Q.K^T)/sqrt(d))
            tensor_softmax_(&self_attention_layer->workspace.attention_scores_scaled[head], 1, &self_attention_layer->workspace.attention_weights[head]);
            bool isnan = tensor_isnan(&self_attention_layer->workspace.attention_weights[head]);
            if(isnan){
                printf("\n\n\nexiting due to nan values in tensor\n");
                exit(1);
            }

            
            // // //  softmax((Q.K^T)/sqrt(d)).V
            tensor_dot_product_(&self_attention_layer->workspace.attention_weights[head], &self_attention_layer->workspace.values_chnuks[head],  &self_attention_layer->workspace.context_vecs[head]);
            // tensor_print(&self_attention_layer->workspace.context_vecs[head], "context_vecs [head]");
            // // printf("======================================================================================\n");

    }

    tensor_concat_(self_attention_layer->workspace.context_vecs, self_attention_layer->n_heads, 1, &self_attention_layer->workspace.concat_heads);
    // tensor_print(&self_attention_layer->workspace.concat_heads, "concat_heads");
    // //tensor_print(&self_attention_layer->params->c_proj.weight, "c_proj");

    // linear_layer_forward(&self_attention_layer->c_proj_layer, &self_attention_layer->workspace.concat_heads);
    // tensor_print(&self_attention_layer->c_proj_layer.workspace.output, "c_proj (Output)");
 }




// void self_attention_layer_print(const SelfAttentionLayer *self_attention_layer, const char *heading){
//     printf("\033[1;31m============================== %s ==============================\033[0m\n", heading);
//     linear_layer_print(&self_attention_layer->W_query, "W_Query");
//     linear_layer_print(&self_attention_layer->W_key, "W_key");
//     linear_layer_print(&self_attention_layer->W_value, "W_value");
//     linear_layer_print(&self_attention_layer->heads_proj, "heads_proj");
// }

