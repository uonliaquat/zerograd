
#include "../../include/layers/transformer.h"
#include "../../include/utils.h"


static void inline transformer_layer_workspace_init(TransformerLayerWorkspace *workspace, char *name){
    char tmp_name[256] = "\0";
    snprintf(tmp_name, sizeof(tmp_name), "%s.resid.1", name);
    tensor_reset(&workspace->resid1_output, tmp_name);
    
    snprintf(tmp_name, sizeof(tmp_name), "%s.resid.2", name);
    tensor_reset(&workspace->resid2_output, tmp_name);
    
    snprintf(tmp_name, sizeof(tmp_name), "%s.output", name);
    tensor_reset(&workspace->output, tmp_name);
}

static void inline transformer_layer_workspace_free(TransformerLayerWorkspace *workspace){
    tensor_free(&workspace->resid1_output);
    tensor_free(&workspace->resid2_output);
    tensor_free(&workspace->output);
}

TransformerLayer transformer_layer_init(TransformerLayerParams *params, const size_t n_heads, char *name){
    TransformerLayer layer;
    layer.n_heads = n_heads;
    strcpy(layer.name, name);

    char tmp_name[256] = "\0";
    snprintf(tmp_name, sizeof(tmp_name), "%s.ln_1", name);
    layer.ln_1  = layer_norm_init(&params->ln_1, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.attn", name);
    layer.attn  = multi_head_attention_layer_init(&params->attn, n_heads, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.ln_2", name);
    layer.ln_2 = layer_norm_init(&params->ln_2, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.mlp", name);
    layer.mlp = multi_layer_perceptron_init(&params->mlp, tmp_name);

    transformer_layer_workspace_init(&layer.workspace, name);
    
    return layer;
}

void transformer_layer_free(TransformerLayer *layer){
    transformer_layer_workspace_free(&layer->workspace);
    layer_norm_free(&layer->ln_1);
    multi_head_attention_layer_free(&layer->attn);
    layer_norm_free(&layer->ln_2);
    multi_layer_perceptron_free(&layer->mlp);

}


void transformer_layer_forward(TransformerLayer *layer, Tensor *x){



    layer_norm_forward(&layer->ln_1, x);
    multi_head_attention_layer_forward(&layer->attn, &layer->ln_1.workspace.output);


    // Resisudal Conenction 1
    tensor_add_(x, &layer->attn.workspace.output, &layer->workspace.resid1_output);

    layer_norm_forward(&layer->ln_2, &layer->workspace.resid1_output);
    multi_layer_perceptron_forward(&layer->mlp, &layer->ln_2.workspace.output);

    // Resisudal Conenction 2
    tensor_add_(&layer->workspace.resid1_output, &layer->mlp.workspace.output, &layer->workspace.resid2_output);
    
    tensor_copy_(&layer->workspace.resid2_output, &layer->workspace.output);
}



void transformer_layer_write(TransformerLayer *layer, Tensor **tensors, size_t *tensors_len){
    layer_norm_write(&layer->ln_1, tensors, tensors_len);
    multi_head_attention_layer_write(&layer->attn, tensors, tensors_len);
    tensors[(*tensors_len)++] = &layer->workspace.resid1_output;
    layer_norm_write(&layer->ln_2, tensors, tensors_len);
    multi_layer_perceptron_write(&layer->mlp, tensors, tensors_len);
    tensors[(*tensors_len)++] = &layer->workspace.resid2_output;
    tensors[(*tensors_len)++] = &layer->workspace.output;
}