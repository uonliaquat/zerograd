
#include "../../include/layers/transformer.h"
#include "../../include/utils.h"


static inline void mlp_workspace_init(MLPWorkspace *workspace){
    workspace->output = (Tensor){0};
}

static inline void mlp_workspace_free(MLPWorkspace *workspace){
    tensor_free(&workspace->output);
}

static inline MLP mlp_init( size_t input_dim, size_t hidden_dim, size_t out_dim, bool bias, bool requires_grad){
    MLP mlp;
    mlp_workspace_init(&mlp.workspace);
    // mlp.layer1 = linear_layer_init(input_dim, hidden_dim, bias, requires_grad, DTYPE_FP32);
    // mlp.layer2 = linear_layer_init(hidden_dim, out_dim, bias, requires_grad, DTYPE_FP32);
    return mlp;
}

static inline void mlp_free(MLP *mlp){
    mlp_workspace_free(&mlp->workspace);
    linear_layer_free(&mlp->layer1);
    linear_layer_free(&mlp->layer2);
}

static inline void mlp_forward(MLP *mlp, Tensor *x){
    print_centered_heading("Feed Forward Network");
    linear_layer_forward(&mlp->layer1,     x);    
    linear_layer_forward(&mlp->layer2,  &mlp->layer1.workspace.output);
}

static inline void transformer_layer_workspace_init(TransformerLayerWorkspace *workspace){
    tensor_reset(&workspace->output);
}

static inline void transformer_layer_workspace_free(TransformerLayerWorkspace *workspace){
    tensor_free(&workspace->output);
}

TransformerLayer transformer_layer_init(TransformerLayerParams *params, const size_t context_len, const size_t emebd_dim, const size_t n_heads, const DataType dtype){
    TransformerLayer transformer_layer;
    transformer_layer_workspace_init(&transformer_layer.workspace);
    transformer_layer.attn_layer = self_attention_layer_init(&params->attn, context_len, emebd_dim, n_heads, dtype);
    // transformer_layer.mlp = mlp_init( emebd_dim, emebd_dim*4, emebd_dim, bias, requires_grad);
    return transformer_layer;
}
void transformer_layer_free(TransformerLayer *transformer_layer){
    mlp_free(&transformer_layer->mlp_layer);
    self_attention_layer_free(&transformer_layer->attn_layer);
}

void transformer_layer_forward(TransformerLayer *transformer_layer, Tensor *x, bool masked){
    print_centered_heading("Self Attention Multi HEAD");
    self_attention_layer_multi_head_forward(&transformer_layer->attn_layer, x, masked);
    // tensor_print(&transformer_layer->self_attention_layer.heads_proj.output, "self_attention_layer heads_proj (Output)");
    // mlp_forward(&transformer_layer->mlp, &transformer_layer->self_attention_layer.heads_proj.output);
    // tensor_print(&transformer_layer->mlp.layer2.output, "Transformer Layer (Output)");
}

void transformer_layer_print(TransformerLayer *transformer_layer, const char *heading){
    print_centered_heading(heading);
}

// void transformer_layer_write(TransformerLayer *transformer_write_fp, const char *base_path){
//     char filename[512] = "\0";
//     snprintf(filename, 512, "%s__%s", base_path, "self_attention_layer");
//     self_attention_layer_write(&transformer_write_fp->self_attention_layer, filename);

//     snprintf(filename, 512, "%s%s", base_path, "__mlp_layer1.csv");
//     // linear_layer_write(&transformer_write_fp->mlp.layer1, filename);

//     snprintf(filename, 512, "%s%s", base_path, "__mlp_layer2.csv");
//     linear_layer_write(&transformer_write_fp->mlp.layer2, filename);
// }