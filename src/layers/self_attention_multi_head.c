
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>


#include "../../include/layers/self_attention.h"
#include "../../include/layers/linear.h"
#include "../../include/utils.h"
#include "../../include/safetensors.h"



static inline void self_attention_layer_workspace_init(SelfAttentionLayer *self_attention_layer, char *name){

    char t_name[256] = "\0";

    snprintf(t_name, sizeof(t_name), "%s.key_transposed", name);
    tensor_reset(&self_attention_layer->workspace.keys_transposed, t_name);

    snprintf(t_name, sizeof(t_name), "%s.attention_scores", name);
    tensor_reset(&self_attention_layer->workspace.attention_scores, t_name);

    snprintf(t_name, sizeof(t_name), "%s.attention_scores_scaled", name);
    tensor_reset(&self_attention_layer->workspace.attention_scores_scaled, t_name);

    snprintf(t_name, sizeof(t_name), "%s.attention_weights", name);
    tensor_reset(&self_attention_layer->workspace.attention_weights, t_name);

    snprintf(t_name, sizeof(t_name), "%s.context_vecs", name);
    tensor_reset(&self_attention_layer->workspace.context_vecs, t_name);

}

static inline void self_attention_layer_workspace_free(const SelfAttentionLayerWorkspace *workspace){
    tensor_free(&workspace->keys_transposed);
    tensor_free(&workspace->attention_scores);
    tensor_free(&workspace->attention_scores_scaled);
    tensor_free(&workspace->attention_weights);
    tensor_free(&workspace->context_vecs);

}

SelfAttentionLayer self_attention_layer_init(const DataType dtype, char *name){
    SelfAttentionLayer self_attention_layer;

    // size_t head_dim                     = embed_dim / n_heads;
    self_attention_layer.dtype = dtype;
    memset(self_attention_layer.name, 0, sizeof(self_attention_layer.name));
    strcpy(self_attention_layer.name, name);

    char l_name[256] = "\0";
    snprintf(l_name, sizeof(l_name), "%s.output", name);
    tensor_reset(&self_attention_layer.output, l_name);
    self_attention_layer_workspace_init(&self_attention_layer, name);
    
    return self_attention_layer;
}


void self_attention_layer_free(const SelfAttentionLayer *self_attention_layer){
    self_attention_layer_workspace_free(&self_attention_layer->workspace);
    tensor_free(&self_attention_layer->output);
}


void self_attention_layer_multi_head_forward(SelfAttentionLayer *self_attention_layer, Tensor *queries, Tensor *keys, Tensor *values, bool masked){
        // // K^T
        tensor_transpose_(keys, &self_attention_layer->workspace.keys_transposed);
        // // Q.K^T
        tensor_dot_product_(
            queries, 
            &self_attention_layer->workspace.keys_transposed, 
            &self_attention_layer->workspace.attention_scores
        );

        // // (Q.K^T)/sqrt(d)
        tensor_elementwise_scale_(
            &self_attention_layer->workspace.attention_scores, 
            1/sqrt(keys->shape[keys->ndim-1]), 
            &self_attention_layer->workspace.attention_scores_scaled
        );    

        if(masked == true){
            tensor_tril_(&self_attention_layer->workspace.attention_scores_scaled, -FLT_MAX); 
        }

        // // softmax((Q.K^T)/sqrt(d))
        tensor_softmax_(&self_attention_layer->workspace.attention_scores_scaled, 1, &self_attention_layer->workspace.attention_weights);

        // // //  softmax((Q.K^T)/sqrt(d)).V
        tensor_dot_product_(&self_attention_layer->workspace.attention_weights, values,  &self_attention_layer->workspace.context_vecs);

        tensor_copy_(&self_attention_layer->workspace.context_vecs, &self_attention_layer->output);
 }

void self_attention_layer_write(SelfAttentionLayer *self_attention_layer, Tensor **tensors, size_t *tensors_len){
    tensors[(*tensors_len)++] = &self_attention_layer->workspace.keys_transposed;
    tensors[(*tensors_len)++] = &self_attention_layer->workspace.attention_scores;
    tensors[(*tensors_len)++] = &self_attention_layer->workspace.attention_scores_scaled;
    tensors[(*tensors_len)++] = &self_attention_layer->workspace.attention_weights;
    tensors[(*tensors_len)++] = &self_attention_layer->workspace.context_vecs;
    tensors[(*tensors_len)++] = &self_attention_layer->output;
}

