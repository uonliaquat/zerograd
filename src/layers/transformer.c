
#include "../../include/layers/transformer.h"
#include "../../include/utils.h"
#include "../../include/safetensors.h"


static inline void mlp_workspace_init(MLPWorkspace *workspace, char *name){

    char l_name[256] = "\0";
    snprintf(l_name, sizeof(l_name), "%s.gelu", name);
    tensor_reset(&workspace->gelu, l_name);
}

static inline void mlp_workspace_free(MLPWorkspace *workspace){
    tensor_free(&workspace->gelu);
}

static inline MLP mlp_init(MLPParams *params, const DataType dtype, char *name){
    MLP mlp;
    mlp.params = params;

    char l_name[256] = "\0";

    snprintf(l_name, sizeof(l_name), "%s.c_fc", name);
    mlp.c_fc =  linear_layer_init(&params->c_fc,   dtype, l_name);

    snprintf(l_name, sizeof(l_name), "%s.c_proj", name);
    mlp.c_proj = linear_layer_init(&params->c_proj, dtype, l_name);

    snprintf(l_name, sizeof(l_name), "%s.output", name);
    tensor_reset(&mlp.output, l_name);

    mlp_workspace_init(&mlp.workspace, name);
    return mlp;
}

static inline void mlp_free(MLP *mlp){
    mlp_workspace_free(&mlp->workspace);
    linear_layer_free(&mlp->c_fc);
    linear_layer_free(&mlp->c_proj);
    tensor_free(&mlp->output);
}



static inline void mlp_forward(MLP *mlp, Tensor *x){
    //print_centered_heading("Feed Forward Network");
    linear_layer_forward(&mlp->c_fc,     x);
    tensor_gelu_(&mlp->c_fc.output, &mlp->workspace.gelu);
    linear_layer_forward(&mlp->c_proj,  &mlp->workspace.gelu);
    tensor_copy_(&mlp->c_proj.output, &mlp->output);
}

static inline void mlp_write(MLP *mlp, Tensor **tensors, size_t *tensors_len){
    linear_layer_write(&mlp->c_fc, tensors, tensors_len);
    tensors[(*tensors_len)++] = &mlp->workspace.gelu;
    linear_layer_write(&mlp->c_proj, tensors, tensors_len);
    tensors[(*tensors_len)++] = &mlp->output;
}

static void inline transformer_layer_workspace_init(TransformerLayerWorkspace *workspace, char *name, size_t n_heads){
    char l_name[256] = "\0";
    snprintf(l_name, sizeof(l_name), "%s.res_out.0", name);
    tensor_reset(&workspace->residual_output[0], l_name);
    
    snprintf(l_name, sizeof(l_name), "%s.res_out.1", name);
    tensor_reset(&workspace->residual_output[1], l_name);

    workspace->qkv = calloc(3, sizeof(Tensor));

    snprintf(l_name, sizeof(l_name), "%s.q", name);
    tensor_reset(&workspace->qkv[0], l_name);

    snprintf(l_name, sizeof(l_name), "%s.k", name);
    tensor_reset(&workspace->qkv[1], l_name);

    snprintf(l_name, sizeof(l_name), "%s.v", name);
    tensor_reset(&workspace->qkv[2], l_name);

    snprintf(l_name, sizeof(l_name), "%s.context_vec", name);
    tensor_reset(&workspace->context_vecs, l_name);

    workspace->queries_heads        = calloc(n_heads, sizeof(Tensor));
    workspace->keys_heads           = calloc(n_heads, sizeof(Tensor));
    workspace->values_heads         = calloc(n_heads, sizeof(Tensor));
    workspace->attn_layer_outputs   = calloc(n_heads, sizeof(Tensor));
    for(size_t i = 0; i < n_heads; i++){
        snprintf(l_name, sizeof(l_name), "%s.q_head.%zu", name, i);
        tensor_reset(&workspace->queries_heads[i], l_name);

        snprintf(l_name, sizeof(l_name), "%s.k_head.%zu", name, i);
        tensor_reset(&workspace->keys_heads[i], l_name);
        
        snprintf(l_name, sizeof(l_name), "%s.v_head.%zu", name, i);
        tensor_reset(&workspace->values_heads[i], l_name);

        snprintf(l_name, sizeof(l_name), "%s.attn.%zu.output", name, i);
        tensor_reset(&workspace->attn_layer_outputs[i], l_name);

    }
}

static void inline transformer_layer_workspace_free(TransformerLayerWorkspace *workspace, size_t n_heads){
    tensor_free(&workspace->residual_output[0]);
    tensor_free(&workspace->residual_output[1]);
    for(size_t i = 0; i < 3; i++){
        tensor_free(&workspace->qkv[i]);
    }

    for(size_t i = 0; i < n_heads; i++){
        tensor_free(&workspace->queries_heads[i]);
        tensor_free(&workspace->keys_heads[i]);
        tensor_free(&workspace->values_heads[i]);
    }
    tensor_free(&workspace->context_vecs);
}

TransformerLayer transformer_layer_init(TransformerLayerParams *params, const size_t context_len, const size_t emebd_dim, const size_t n_heads, const bool masked, const DataType dtype, char *name){
    TransformerLayer transformer_layer;
    transformer_layer.n_heads = n_heads;
    transformer_layer.masked = masked;
    char l_name[256] = "\0";
    snprintf(l_name, sizeof(l_name), "%s.ln_0", name);
    transformer_layer.ln_1_layer  = layer_norm_init(&params->ln_1, dtype, l_name);

    snprintf(l_name, sizeof(l_name), "%s.ln_1", name);
    transformer_layer.ln_2_layer   = layer_norm_init(&params->ln_2, dtype, l_name);

    snprintf(l_name, sizeof(l_name), "%s.c_attn", name);
    transformer_layer.c_attn_layer  = linear_layer_init(&params->c_attn, dtype, l_name);

    snprintf(l_name, sizeof(l_name), "%s.c_proj", name);
    transformer_layer.c_proj_layer = linear_layer_init(&params->c_proj, dtype, l_name);


    // // transformer_layer.attn_layer   = calloc(n_heads, sizeof(SelfAttentionLayer));
    for(size_t i = 0; i < n_heads; i++){
        snprintf(l_name, sizeof(l_name), "%s.attn.%zu", name, i);
        transformer_layer.attn_layer[i]   = self_attention_layer_init(dtype, l_name);
    }
    
    snprintf(l_name, sizeof(l_name), "%s.mlp", name);
    transformer_layer.mlp_layer     = mlp_init(&params->mlp, dtype, l_name);


    transformer_layer_workspace_init(&transformer_layer.workspace, name, n_heads);
    
    snprintf(l_name, sizeof(l_name), "%s.output", name);
    tensor_reset(&transformer_layer.output, l_name);
    return transformer_layer;
}
void transformer_layer_free(TransformerLayer *transformer_layer){
    transformer_layer_workspace_free(&transformer_layer->workspace, transformer_layer->n_heads);
    layer_norm_free(&transformer_layer->ln_1_layer);
    layer_norm_free(&transformer_layer->ln_2_layer);
    linear_layer_free(&transformer_layer->c_attn_layer);
    linear_layer_free(&transformer_layer->c_proj_layer);
    mlp_free(&transformer_layer->mlp_layer);
    tensor_free(&transformer_layer->output);

    for(size_t i = 0; i < transformer_layer->n_heads; i++){
        self_attention_layer_free(&transformer_layer->attn_layer[i]);
    }
}

void transformer_layer_forward(TransformerLayer *transformer_layer, Tensor *x){
    //print_centered_heading("Self Attention Multi HEAD");
    layer_norm_forward(&transformer_layer->ln_1_layer, x);
    linear_layer_forward(&transformer_layer->c_attn_layer, &transformer_layer->ln_1_layer.output);

    tensor_chunk_(&transformer_layer->c_attn_layer.output, 3, 1, transformer_layer->workspace.qkv);
    tensor_chunk_(&transformer_layer->workspace.qkv[0],   transformer_layer->n_heads, 1, transformer_layer->workspace.queries_heads);
    tensor_chunk_(&transformer_layer->workspace.qkv[1],   transformer_layer->n_heads, 1, transformer_layer->workspace.keys_heads);
    tensor_chunk_(&transformer_layer->workspace.qkv[2],   transformer_layer->n_heads, 1, transformer_layer->workspace.values_heads);
    
    for(size_t head = 0; head < transformer_layer->n_heads; head++){
        self_attention_layer_multi_head_forward(
            &transformer_layer->attn_layer[head], 
            &transformer_layer->workspace.queries_heads[head], 
            &transformer_layer->workspace.keys_heads[head], 
            &transformer_layer->workspace.values_heads[head], 
            transformer_layer->masked
        );
        tensor_copy_(&transformer_layer->attn_layer[head].output, &transformer_layer->workspace.attn_layer_outputs[head]);
    }
    tensor_concat_(transformer_layer->workspace.attn_layer_outputs, transformer_layer->n_heads, 1, &transformer_layer->workspace.context_vecs);  
  
    linear_layer_forward(&transformer_layer->c_proj_layer,  &transformer_layer->workspace.context_vecs);


    //Residual Connection
    tensor_add_(x, &transformer_layer->c_proj_layer.output, &transformer_layer->workspace.residual_output[0]);

    layer_norm_forward(&transformer_layer->ln_2_layer, &transformer_layer->workspace.residual_output[0]);

    mlp_forward(&transformer_layer->mlp_layer, &transformer_layer->ln_2_layer.output);

    tensor_add_(&transformer_layer->workspace.residual_output[0], &transformer_layer->mlp_layer.output, &transformer_layer->workspace.residual_output[1]);

    tensor_copy_(&transformer_layer->workspace.residual_output[1], &transformer_layer->output);
}

void transformer_layer_print(TransformerLayer *transformer_layer, const char *heading){
    print_centered_heading(heading);
}

void transformer_layer_write(TransformerLayer *transforemr_layer, Tensor **tensors, size_t *tensors_len){
    layer_norm_write(&transforemr_layer->ln_1_layer, tensors, tensors_len);
    //layer_norm_write(&transforemr_layer->ln_2_layer, tensors, tensors_len);

    linear_layer_write(&transforemr_layer->c_attn_layer, tensors, tensors_len);
    tensors[(*tensors_len)++] = &transforemr_layer->workspace.qkv[0];
    tensors[(*tensors_len)++] = &transforemr_layer->workspace.qkv[1];
    tensors[(*tensors_len)++] = &transforemr_layer->workspace.qkv[2];

    for(size_t i = 0; i < transforemr_layer->n_heads; i++){
        tensors[(*tensors_len)++] = &transforemr_layer->workspace.queries_heads[i];
        tensors[(*tensors_len)++] = &transforemr_layer->workspace.keys_heads[i];
        tensors[(*tensors_len)++] = &transforemr_layer->workspace.values_heads[i];
        self_attention_layer_write(&transforemr_layer->attn_layer[i], tensors, tensors_len);
    }
    tensors[(*tensors_len)++] = &transforemr_layer->workspace.context_vecs;

    linear_layer_write(&transforemr_layer->c_proj_layer, tensors, tensors_len);

    tensors[(*tensors_len)++] = &transforemr_layer->workspace.residual_output[0];
    layer_norm_write(&transforemr_layer->ln_2_layer, tensors, tensors_len);

    mlp_write(&transforemr_layer->mlp_layer, tensors, tensors_len);

    tensors[(*tensors_len)++] = &transforemr_layer->workspace.residual_output[1];
    tensors[(*tensors_len)++] = &transforemr_layer->output;

}