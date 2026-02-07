#include <stdlib.h>
#include <string.h>
#include "../../include/layers/layer_norm.h"



static inline void layer_norm_params_free(LayerNormParams *params){
    tensor_free(&params->bias);
    tensor_free(&params->weight);
}


static inline void layer_norm_workspace_init(LayerNormWorkSpace *workspace){
    tensor_reset(&workspace->mean_var);
    tensor_reset(&workspace->x_norm);
    tensor_reset(&workspace->x_norm_scaled);
}

static inline void layer_norm_workspace_free(LayerNormWorkSpace *workspace){
    tensor_free(&workspace->mean_var);
    tensor_free(&workspace->x_norm);
    tensor_free(&workspace->x_norm_scaled);
}

LayerNorm layer_norm_init(LayerNormParams *params, const DataType dtype){
    LayerNorm layer_norm;
    layer_norm.params = params;
    layer_norm.eps = 1e-9;
    tensor_reset(&layer_norm.output);
    layer_norm_workspace_init(&layer_norm.workspace);
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
    tensor_vector_scale_(&layer_norm->workspace.x_norm, &layer_norm->params->weight, &layer_norm->output);
    tensor_vector_add_(&layer_norm->output, &layer_norm->params->bias, &layer_norm->output);
}