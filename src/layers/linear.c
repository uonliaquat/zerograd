#include "../../include/layers/linear.h"
#include "../../include/safetensors.h"
#include <stdlib.h>
#include <string.h>


static inline void linear_layer_workspace_init(LinearLayerWorkspace *workspace, char *name){

    char tmp_name[128] = "\0";
    snprintf(tmp_name, sizeof(tmp_name), "%s.a", name);
    tensor_reset(&workspace->a, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s", name);
    tensor_reset(&workspace->output, tmp_name);
}

static inline void linear_layer_workspace_free(const LinearLayerWorkspace *workspace){
    tensor_free(&workspace->a);
    tensor_free(&workspace->output);
}

static inline void linear_layer_params_free(const LinearLayerParams *params){
    tensor_free(&params->weight);
    tensor_free(&params->bias);
}


LinearLayer linear_layer_init(LinearLayerParams *params, char *name){
    LinearLayer layer;
    strcpy(layer.name, name);
    layer.params = params;
    linear_layer_workspace_init(&layer.workspace, name);
    return layer;
}

void linear_layer_free(const LinearLayer *layer){
    linear_layer_workspace_free(&layer->workspace);
    linear_layer_params_free(layer->params);
}


void linear_layer_forward(LinearLayer *layer, Tensor *x){
    // Y = X.W
    tensor_dot_product_(x, &layer->params->weight, &layer->workspace.a);
    tensor_vector_add_(&layer->workspace.a, &layer->params->bias, &layer->workspace.output);
}


void linear_layer_write(LinearLayer *layer, Tensor **tensors, size_t *tensors_len){
    tensors[(*tensors_len)++] = &layer->workspace.a;
    tensors[(*tensors_len)++] = &layer->workspace.output;
}
