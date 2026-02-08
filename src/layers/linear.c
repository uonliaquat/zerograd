#include "../../include/layers/linear.h"
#include "../../include/safetensors.h"
#include <stdlib.h>
#include <string.h>
// LinearLayer linear_layer_init(const size_t inputs, const size_t outputs, const bool bias, const bool requires_grad, const DataType dtype){
//     LinearLayer linear_layer;
//     linear_layer.weights  = tensor_init(NULL, (uint32_t[]){inputs, outputs}, dtype, NULL);
//     linear_layer.has_bias = bias;
//     if(bias == true)
//         linear_layer.bias = tensor_init(NULL, (uint32_t[]){1, outputs}, dtype, NULL);
//     linear_layer.dtype = dtype;
//     linear_layer.output = (Tensor){0};
//     return linear_layer;
// }

static inline void linear_layer_workspace_init(LinearLayerWorkspace *workspace, char *name){

    char l_name[256] = "\0";
    snprintf(l_name, sizeof(l_name), "%s.a", name);
    tensor_reset(&workspace->a, l_name);
}

static inline void linear_layer_workspace_free(const LinearLayerWorkspace *workspace){
    tensor_free(&workspace->a);
}


LinearLayer linear_layer_init(LinearLayerParams *params, const DataType dtype, char *name){
    LinearLayer linear_layer;

    linear_layer.params  = params;
    linear_layer.dtype = dtype;
    memset(linear_layer.name, 0, sizeof(linear_layer.name));
    strcpy(linear_layer.name, name);
    char l_name[256] = "\0";
    snprintf(l_name, sizeof(l_name), "%s.output", name);
    tensor_reset(&linear_layer.output, l_name);

    linear_layer_workspace_init(&linear_layer.workspace, name);
    return linear_layer;
}

void linear_layer_params_free(const LinearLayerParams *params){
    tensor_free(&params->weight);
    tensor_free(&params->bias);
}

void linear_layer_free(const LinearLayer *linear_layer){
    linear_layer_workspace_free(&linear_layer->workspace);
    tensor_free(&linear_layer->output);
    linear_layer_params_free(linear_layer->params);
}


void linear_layer_forward(LinearLayer *linear_layer, Tensor *x){
    // Y = X.W
    tensor_dot_product_(x, &linear_layer->params->weight, &linear_layer->workspace.a);
    //tensor_print(&linear_layer->workspace.a, "linear_layer->workspace.a");
    tensor_vector_add_(&linear_layer->workspace.a, &linear_layer->params->bias, &linear_layer->output);
    //tensor_print(&linear_layer->output, "linear_layer->output");
}


void linear_layer_write(LinearLayer *linear_layer, Tensor **tensors, size_t *tensors_len){
    tensors[(*tensors_len)++] = &linear_layer->workspace.a;
    tensors[(*tensors_len)++] = &linear_layer->output;
}
