#include <stdlib.h>
#include <string.h>
#include "../../include/safetensors.h"
// #include "../../include/layers/layer_norm.h"



static inline void layer_norm_params_free(LayerNormParams *params){
    tensor_free(&params->bias);
    tensor_free(&params->weight);
}


static inline void layer_norm_workspace_init(LayerNormWorkSpace *workspace, char *name){
    char l_name[256] = "\0";

    snprintf(l_name, sizeof(l_name), "%s.mean_var", name);
    tensor_reset(&workspace->mean_var,          l_name);

    snprintf(l_name, sizeof(l_name), "%s.x_norm", name);
    tensor_reset(&workspace->x_norm,            l_name);

    snprintf(l_name, sizeof(l_name), "%s.x_norm_scaled", name);
    tensor_reset(&workspace->x_norm_scaled,     l_name);

    snprintf(l_name, sizeof(l_name), "%s.x_norm_shifted", name);
    tensor_reset(&workspace->x_norm_shifted,    l_name);
}

static inline void layer_norm_workspace_free(LayerNormWorkSpace *workspace){
    tensor_free(&workspace->mean_var);
    tensor_free(&workspace->x_norm);
    tensor_free(&workspace->x_norm_scaled);
    tensor_free(&workspace->x_norm_shifted);
}

LayerNorm layer_norm_init(LayerNormParams *params, const DataType dtype, char *name){
    LayerNorm layer_norm;
    layer_norm.params = params;
    layer_norm.eps = 1e-5;
    memset(layer_norm.name, 0, sizeof(layer_norm.name));
    strcpy(layer_norm.name, name);

    char l_name[256] = "\0";
    snprintf(l_name, sizeof(l_name), "%s.output", name);
    tensor_reset(&layer_norm.output, l_name);
    layer_norm_workspace_init(&layer_norm.workspace, name);
    return layer_norm;
}

void layer_norm_free(LayerNorm *layer_norm){
    layer_norm_workspace_free(&layer_norm->workspace);
    tensor_free(&layer_norm->output);
    layer_norm_params_free(layer_norm->params);
}

void layer_norm_forward(LayerNorm *layer_norm, Tensor *x){
    tensor_mean_var_(x, &layer_norm->workspace.mean_var);
    tensor_norm_(x, &layer_norm->workspace.mean_var, layer_norm->eps, &layer_norm->workspace.x_norm);
    tensor_vector_scale_(&layer_norm->workspace.x_norm, &layer_norm->params->weight, &layer_norm->workspace.x_norm_scaled);
    tensor_vector_add_(&layer_norm->workspace.x_norm_scaled, &layer_norm->params->bias, &layer_norm->workspace.x_norm_shifted);
    tensor_copy_(&layer_norm->workspace.x_norm_shifted, &layer_norm->output);

    // tensor_print(&layer_norm->workspace.mean_var, "&layer_norm->workspace.mean_var");
    // tensor_print(&layer_norm->workspace.x_norm, "&layer_norm->workspace.x_norm");
    // tensor_print(&layer_norm->workspace.x_norm_scaled, "&layer_norm->workspace.x_norm_scaled");
    // tensor_print(&layer_norm->output, "layer_norm.output");
}

void layer_norm_write(LayerNorm *layer_norm, Tensor **tensors, size_t *tensors_len){
    tensors[(*tensors_len)++] = &layer_norm->workspace.mean_var;
    tensors[(*tensors_len)++] = &layer_norm->workspace.x_norm;
    tensors[(*tensors_len)++] = &layer_norm->workspace.x_norm_scaled;
    tensors[(*tensors_len)++] = &layer_norm->workspace.x_norm_shifted;
    tensors[(*tensors_len)++] = &layer_norm->output;
}