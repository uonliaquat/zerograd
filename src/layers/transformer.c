
#include "../../include/layers/transformer.h"
#include "../../include/utils.h"



static inline MLP mlp_init(MLPParams *params, const DataType dtype){
    MLP mlp;
    mlp.params = params;
    mlp.layer1 = linear_layer_init(&params->c_fc, dtype);
    mlp.layer2 = linear_layer_init(&params->c_proj, dtype);
    tensor_reset(&mlp.output);
    return mlp;
}

static inline void mlp_free(MLP *mlp){
    linear_layer_free(&mlp->layer1);
    linear_layer_free(&mlp->layer2);
    tensor_free(&mlp->output);
}

static inline void mlp_forward(MLP *mlp, Tensor *x){
    print_centered_heading("Feed Forward Network");
    linear_layer_forward(&mlp->layer1,     x);    
    linear_layer_forward(&mlp->layer2,  &mlp->layer1.output);
}


static void inline transformer_layer_workspace_init(TransformerLayerWorkspace *workspace){
    tensor_reset(&workspace->residual_output);
}

static void inline transformer_layer_workspace_free(TransformerLayerWorkspace *workspace){
    tensor_free(&workspace->residual_output);
}

TransformerLayer transformer_layer_init(TransformerLayerParams *params, const size_t context_len, const size_t emebd_dim, const size_t n_heads, const bool masked, const DataType dtype){
    TransformerLayer transformer_layer;
    transformer_layer.masked = masked;
    transformer_layer.attn_layer    = self_attention_layer_init(&params->attn, context_len, emebd_dim, n_heads, dtype);
    transformer_layer.mlp_layer     = mlp_init(&params->mlp, dtype);
    transformer_layer.ln_layer[0]   = layer_norm_init(&params->ln_[0], dtype);
    transformer_layer.ln_layer[1]   = layer_norm_init(&params->ln_[1], dtype);

    transformer_layer_workspace_init(&transformer_layer.workspace);
    return transformer_layer;
}
void transformer_layer_free(TransformerLayer *transformer_layer){
    transformer_layer_workspace_free(&transformer_layer->workspace);
    layer_norm_free(&transformer_layer->ln_layer[0]);
    layer_norm_free(&transformer_layer->ln_layer[1]);
    self_attention_layer_free(&transformer_layer->attn_layer);
    mlp_free(&transformer_layer->mlp_layer);

}

void transformer_layer_forward(TransformerLayer *transformer_layer, Tensor *x){
    print_centered_heading("Self Attention Multi HEAD");
    layer_norm_forward(&transformer_layer->ln_layer[0], x);
    tensor_print(&transformer_layer->ln_layer[0].output, "layer_norm_0 (output)");
    
    self_attention_layer_multi_head_forward(&transformer_layer->attn_layer, &transformer_layer->ln_layer[0].output, transformer_layer->masked);
    tensor_print(&transformer_layer->attn_layer.output, "Tranformer Layer (Output)");

    //Residual Connection
    tensor_add_(x, &transformer_layer->attn_layer.output, &transformer_layer->workspace.residual_output);
    tensor_print(&transformer_layer->workspace.residual_output, "residual (output)");

    layer_norm_forward(&transformer_layer->ln_layer[1], &transformer_layer->attn_layer.output);
    tensor_print(&transformer_layer->ln_layer[1].output, "layer_norm_1 (output)");

    //mlp_forward(&transformer_layer->mlp_layer, &transformer_layer->attn_layer.output);
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