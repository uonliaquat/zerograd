#include <stdlib.h>
#include <string.h>

#include "../../include/layers/layer_norm.h"



static inline void layer_norm_workspace_init(LayerNormWorkSpace *workspace, char *name){
    char tmp_name[128] = "\0";

    snprintf(tmp_name, sizeof(tmp_name), "%s.mean_var", name);
    tensor_reset(&workspace->mean_var,          tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.x_norm", name);
    tensor_reset(&workspace->x_norm,            tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.x_norm_scaled", name);
    tensor_reset(&workspace->x_norm_scaled,     tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.x_norm_shifted", name);
    tensor_reset(&workspace->x_norm_shifted,    tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s", name);
    tensor_reset(&workspace->output, tmp_name);
}

static inline void layer_norm_workspace_free(LayerNormWorkSpace *workspace){
    tensor_free(&workspace->mean_var);
    tensor_free(&workspace->x_norm);
    tensor_free(&workspace->x_norm_scaled);
    tensor_free(&workspace->x_norm_shifted);
}

static inline void layer_norm_params_free(LayerNormParams *params){
    tensor_free(&params->weight);
    tensor_free(&params->bias);
}

LayerNorm layer_norm_init(LayerNormParams *params, char *name){
    LayerNorm layer;
    layer.eps = 1e-5;
    layer.params = params;
    strcpy(layer.name, name);

    layer_norm_workspace_init(&layer.workspace, name);
    return layer;
}

void layer_norm_free(LayerNorm *layer){
    layer_norm_workspace_free(&layer->workspace);
    layer_norm_params_free(layer->params);
}

void layer_norm_forward(LayerNorm *layer_norm, Tensor *x){
    tensor_mean_var_(x, &layer_norm->workspace.mean_var);
    tensor_norm_(x, &layer_norm->workspace.mean_var, layer_norm->eps, &layer_norm->workspace.x_norm);
    tensor_vector_scale_(&layer_norm->workspace.x_norm, &layer_norm->params->weight, &layer_norm->workspace.x_norm_scaled);
    tensor_vector_add_(&layer_norm->workspace.x_norm_scaled, &layer_norm->params->bias, &layer_norm->workspace.x_norm_shifted);
    tensor_copy_(&layer_norm->workspace.x_norm_shifted, &layer_norm->workspace.output);

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
    tensors[(*tensors_len)++] = &layer_norm->workspace.output;
}