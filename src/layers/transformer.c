
#include "../../include/layers/transformer.h"
#include "../../include/utils.h"


static inline void mlp_workspace_init(MLPWorkspace *workspace){
    tensor_reset(&workspace->gelu);
}

static inline void mlp_workspace_free(MLPWorkspace *workspace){
    tensor_free(&workspace->gelu);
}

static inline MLP mlp_init(MLPParams *params, const DataType dtype){
    MLP mlp;
    mlp.params = params;
    mlp.layer1 = linear_layer_init(&params->c_fc, dtype);
    mlp.layer2 = linear_layer_init(&params->c_proj, dtype);
    tensor_reset(&mlp.output);
    mlp_workspace_init(&mlp.workspace);
    return mlp;
}

static inline void mlp_free(MLP *mlp){
    mlp_workspace_free(&mlp->workspace);
    linear_layer_free(&mlp->layer1);
    linear_layer_free(&mlp->layer2);
    tensor_free(&mlp->output);
}



static inline void mlp_forward(MLP *mlp, Tensor *x){
    //print_centered_heading("Feed Forward Network");
    linear_layer_forward(&mlp->layer1,     x);
    tensor_gelu_(&mlp->layer1.output, &mlp->workspace.gelu);
    linear_layer_forward(&mlp->layer2,  &mlp->workspace.gelu);
    tensor_copy_(&mlp->layer2.output, &mlp->output);
}

static void inline transformer_layer_workspace_init(TransformerLayerWorkspace *workspace){
    tensor_reset(&workspace->residual_output[0]);
    tensor_reset(&workspace->residual_output[1]);
}

static void inline transformer_layer_workspace_free(TransformerLayerWorkspace *workspace){
    tensor_free(&workspace->residual_output[0]);
    tensor_free(&workspace->residual_output[1]);
}

TransformerLayer transformer_layer_init(TransformerLayerParams *params, const size_t context_len, const size_t emebd_dim, const size_t n_heads, const bool masked, const DataType dtype){
    TransformerLayer transformer_layer;
    transformer_layer.masked = masked;
    transformer_layer.attn_layer    = self_attention_layer_init(&params->attn, context_len, emebd_dim, n_heads, dtype);
    transformer_layer.mlp_layer     = mlp_init(&params->mlp, dtype);
    transformer_layer.ln_layer[0]   = layer_norm_init(&params->ln_[0], dtype);
    transformer_layer.ln_layer[1]   = layer_norm_init(&params->ln_[1], dtype);

    transformer_layer_workspace_init(&transformer_layer.workspace);
    tensor_reset(&transformer_layer.output);
    return transformer_layer;
}
void transformer_layer_free(TransformerLayer *transformer_layer){
    transformer_layer_workspace_free(&transformer_layer->workspace);
    layer_norm_free(&transformer_layer->ln_layer[0]);
    layer_norm_free(&transformer_layer->ln_layer[1]);
    self_attention_layer_free(&transformer_layer->attn_layer);
    mlp_free(&transformer_layer->mlp_layer);
    tensor_free(&transformer_layer->output);
}

void transformer_layer_forward(TransformerLayer *transformer_layer, Tensor *x){
    //print_centered_heading("Self Attention Multi HEAD");
    layer_norm_forward(&transformer_layer->ln_layer[0], x);
    tensor_print(&transformer_layer->ln_layer[0].output, "layer_norm_0 (output)");

    if(tensor_isnan(&transformer_layer->ln_layer[0].output)){
        printf("\n\n\nexiting due to nan values in , &transformer_layer->ln_layer[0].output\n");
        exit(1);
    }

    
    self_attention_layer_multi_head_forward(&transformer_layer->attn_layer, &transformer_layer->ln_layer[0].output, transformer_layer->masked);
    tensor_print(&transformer_layer->attn_layer.output, "Self Attention Layer (Output)");

    if(tensor_isnan(&transformer_layer->attn_layer.output)){
        printf("\n\n\nexiting due to nan values in , &transformer_layer->attn_layer.output\n");
        exit(1);
    }

    //Residual Connection
    tensor_add_(x, &transformer_layer->attn_layer.output, &transformer_layer->workspace.residual_output[0]);
    tensor_print(&transformer_layer->workspace.residual_output[0], "residual_connection_0 (output)");

    if(tensor_isnan(&transformer_layer->workspace.residual_output[0])){
        printf("\n\n\nexiting due to nan values in , &transformer_layer->workspace.residual_output[0]\n");
        exit(1);
    }

    layer_norm_forward(&transformer_layer->ln_layer[1], &transformer_layer->workspace.residual_output[0]);
    tensor_print(&transformer_layer->ln_layer[1].output, "layer_norm_1 (output)");

    if(tensor_isnan(&transformer_layer->ln_layer[1].output)){
        printf("\n\n\nexiting due to nan values in , &transformer_layer->ln_layer[1].output\n");
        exit(1);
    }

    mlp_forward(&transformer_layer->mlp_layer, &transformer_layer->ln_layer[1].output);
    tensor_print(&transformer_layer->mlp_layer.output, "MLP Layer (Output)");

    if(tensor_isnan(&transformer_layer->mlp_layer.output)){
        printf("\n\n\nexiting due to nan values in , &transformer_layer->mlp_layer.output\n");
        exit(1);
    }


    tensor_add_(&transformer_layer->attn_layer.output, &transformer_layer->mlp_layer.output, &transformer_layer->workspace.residual_output[1]);

    if(tensor_isnan(&transformer_layer->workspace.residual_output[1])){
        printf("\n\n\nexiting due to nan values in , &transformer_layer->workspace.residual_output[1]\n");
        exit(1);
    }

    tensor_copy_(&transformer_layer->workspace.residual_output[1], &transformer_layer->output);
    if(tensor_isnan(&transformer_layer->output)){
        printf("\n\n\nexiting due to nan values in , &transformer_layer->output\n");
        exit(1);
    }


    tensor_print(&transformer_layer->output, "residual_connection_1 (output)");
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