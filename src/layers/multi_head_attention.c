
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>


#include "../../include/layers/multi_head_attention.h"
#include "../../include/layers/linear.h"
#include "../../include/utils.h"
// #include "../../include/safetensors.h"



static inline void self_attention_layer_workspace_init(MultiHeadAttentionLayerWorkspace *workspace, const size_t n_heads, char *name){

    
    workspace->q_heads              = calloc(n_heads, sizeof(Tensor));
    workspace->k_heads              = calloc(n_heads, sizeof(Tensor));
    workspace->v_heads              = calloc(n_heads, sizeof(Tensor));
    workspace->keys_transposed      = calloc(n_heads, sizeof(Tensor));
    workspace->attn_scores          = calloc(n_heads, sizeof(Tensor));
    workspace->attn_scores_scaled   = calloc(n_heads, sizeof(Tensor));
    workspace->attn_scores_masked   = calloc(n_heads, sizeof(Tensor));
    workspace->attn_weights         = calloc(n_heads, sizeof(Tensor));
    workspace->ctx_vecs             = calloc(n_heads, sizeof(Tensor));

    char tmp_name[128] = "\0";
    snprintf(tmp_name, sizeof(tmp_name), "%s.q", name);
    tensor_reset(&workspace->qkv[0], tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.k", name);
    tensor_reset(&workspace->qkv[1], tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.v", name);
    tensor_reset(&workspace->qkv[2], tmp_name);

    for(size_t head = 0; head < n_heads; head++){

        snprintf(tmp_name, sizeof(tmp_name), "%s.%zu.q_head", name, head);
        tensor_reset(&workspace->q_heads[head], tmp_name);

        snprintf(tmp_name, sizeof(tmp_name), "%s.%zu.k_head", name, head);
        tensor_reset(&workspace->k_heads[head], tmp_name);

        snprintf(tmp_name, sizeof(tmp_name), "%s.%zu.v_head", name, head);
        tensor_reset(&workspace->v_heads[head], tmp_name);

        snprintf(tmp_name, sizeof(tmp_name), "%s.%zu.key_transposed", name, head);
        tensor_reset(&workspace->keys_transposed[head], tmp_name);

        snprintf(tmp_name, sizeof(tmp_name), "%s.%zu.attn_score", name, head);
        tensor_reset(&workspace->attn_scores[head], tmp_name);

        snprintf(tmp_name, sizeof(tmp_name), "%s.%zu.attn_score_scaled", name, head);
        tensor_reset(&workspace->attn_scores_scaled[head], tmp_name);

        snprintf(tmp_name, sizeof(tmp_name), "%s.%zu.attn_score_masked", name, head);
        tensor_reset(&workspace->attn_scores_masked[head], tmp_name);

        snprintf(tmp_name, sizeof(tmp_name), "%s.%zu.attn_weight", name, head);
        tensor_reset(&workspace->attn_weights[head], tmp_name);

        snprintf(tmp_name, sizeof(tmp_name), "%s.%zu.ctx_vec", name, head);
        tensor_reset(&workspace->ctx_vecs[head], tmp_name);
    }

    snprintf(tmp_name, sizeof(tmp_name), "%s.concat_vecs", name);
    tensor_reset(&workspace->conact_vecs, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.output", name);
    tensor_reset(&workspace->output, tmp_name);

}

static inline void self_attention_layer_workspace_free(const MultiHeadAttentionLayerWorkspace *workspace, const size_t n_heads){
    
    tensor_free(&workspace->qkv[0]);
    tensor_free(&workspace->qkv[1]);
    tensor_free(&workspace->qkv[2]);

    for(size_t head = 0; head < n_heads; head++){
        tensor_free(&workspace->q_heads[head]);
        tensor_free(&workspace->k_heads[head]);
        tensor_free(&workspace->v_heads[head]);
        tensor_free(&workspace->keys_transposed[head]);
        tensor_free(&workspace->attn_scores[head]);
        tensor_free(&workspace->attn_scores_scaled[head]);
        tensor_free(&workspace->attn_scores_masked[head]);
        tensor_free(&workspace->attn_weights[head]);
        tensor_free(&workspace->ctx_vecs[head]);
    }
    tensor_free(&workspace->conact_vecs);
    tensor_free(&workspace->output);
}


MultiHeadAttentionLayer multi_head_attention_layer_init(MultiHeadAttentionLayerParams *params, const size_t n_heads, char *name){
    MultiHeadAttentionLayer layer;
    layer.n_heads = n_heads;
    tensor_copy_(&layer.bias, &params->bias);

    char tmp_name[128] = "\0";
    snprintf(tmp_name, sizeof(tmp_name), "%s.c_attn", name);
    layer.c_attn = linear_layer_init(&params->c_attn, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.c_proj", name);
    layer.c_proj = linear_layer_init(&params->c_proj, tmp_name);

    // size_t head_dim                     = embed_dim / n_heads;
    self_attention_layer_workspace_init(&layer.workspace, n_heads, name);
    
    return layer;
}


void multi_head_attention_layer_free(const MultiHeadAttentionLayer *layer){
    self_attention_layer_workspace_free(&layer->workspace, layer->n_heads);
    linear_layer_free(&layer->c_attn);
    linear_layer_free(&layer->c_proj);
    tensor_free(&layer->bias);
}


void multi_head_attention_layer_forward(MultiHeadAttentionLayer *layer, Tensor *x){
    
    linear_layer_forward(&layer->c_attn, x);

    tensor_chunk_(&layer->c_attn.workspace.output, 3, 1, layer->workspace.qkv);
    tensor_chunk_(&layer->workspace.qkv[0],   layer->n_heads, 1, layer->workspace.q_heads);
    tensor_chunk_(&layer->workspace.qkv[1],   layer->n_heads, 1, layer->workspace.k_heads);
    tensor_chunk_(&layer->workspace.qkv[2],   layer->n_heads, 1, layer->workspace.v_heads);

    for(size_t head = 0; head < layer->n_heads; head++){
        // // K^T
        tensor_transpose_(&layer->workspace.k_heads[head], &layer->workspace.keys_transposed[head]);

        // // Q.K^T
        tensor_dot_product_(
            &layer->workspace.q_heads[head], 
            &layer->workspace.keys_transposed[head], 
            &layer->workspace.attn_scores[head]
        );

        // // (Q.K^T)/sqrt(d)
        size_t embed_dim = layer->workspace.k_heads[head].shape[layer->workspace.k_heads[head].ndim-1];

        tensor_elementwise_scale_(
            &layer->workspace.attn_scores[head], 
            1/sqrt(embed_dim), 
            &layer->workspace.attn_scores_scaled[head]
        );    

        //tensor_print(layer->bias, "layer->bias");
        //tensor_print(&layer->workspace.attn_scores_scaled[head], "&layer->workspace.attn_scores_scaled[head]");
        //tensor_add_(&layer->workspace.attn_scores_scaled[head], layer->bias, &layer->workspace.attn_scores_masked[head]);
        tensor_tril_(&layer->workspace.attn_scores_scaled[head], -FLT_MAX); 

        // // softmax((Q.K^T)/sqrt(d))
        tensor_softmax_(
            &layer->workspace.attn_scores_scaled[head],
            1,
            &layer->workspace.attn_weights[head]
        );

        // // //  softmax((Q.K^T)/sqrt(d)).V
        tensor_dot_product_(
            &layer->workspace.attn_weights[head], 
            &layer->workspace.v_heads[head], 
            &layer->workspace.ctx_vecs[head]
        );
        // tensor_copy_(&layer->workspace.ctx_vecs[head], &layer->output);
    }
    tensor_concat_(layer->workspace.ctx_vecs, layer->n_heads, 1, &layer->workspace.conact_vecs);  
    linear_layer_forward(&layer->c_proj,  &layer->workspace.conact_vecs);
    tensor_copy_(&layer->c_proj.workspace.output, &layer->workspace.output);
 }

void multi_head_attention_layer_write(MultiHeadAttentionLayer *layer, Tensor **tensors, size_t *tensors_len){
    linear_layer_write(&layer->c_attn, tensors, tensors_len);
    tensors[(*tensors_len)++] = &layer->workspace.qkv[0];
    tensors[(*tensors_len)++] = &layer->workspace.qkv[1];
    tensors[(*tensors_len)++] = &layer->workspace.qkv[2];
    for(size_t head = 0; head < layer->n_heads; head++){
        tensors[(*tensors_len)++] = &layer->workspace.q_heads[head];
        tensors[(*tensors_len)++] = &layer->workspace.k_heads[head];
        tensors[(*tensors_len)++] = &layer->workspace.v_heads[head];
        tensors[(*tensors_len)++] = &layer->workspace.keys_transposed[head];
        tensors[(*tensors_len)++] = &layer->workspace.attn_scores[head];
        tensors[(*tensors_len)++] = &layer->workspace.attn_scores_scaled[head];
        // tensors[(*tensors_len)++] = &layer->workspace.attn_scores_masked[head];
        tensors[(*tensors_len)++] = &layer->workspace.attn_weights[head];
        tensors[(*tensors_len)++] = &layer->workspace.ctx_vecs[head];
    }
    tensors[(*tensors_len)++] = &layer->workspace.conact_vecs;
    tensors[(*tensors_len)++] = &layer->workspace.output;
    linear_layer_write(&layer->c_proj, tensors, tensors_len);
}

