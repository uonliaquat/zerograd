#include "../../include/layers/linear.h"

#include <stdlib.h>
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

static inline void linear_layer_workspace_init(LinearLayerWorkspace *workspace){
    tensor_reset(&workspace->a);
}

static inline void linear_layer_workspace_free(const LinearLayerWorkspace *workspace){
    tensor_free(&workspace->a);
}


LinearLayer linear_layer_init(LinearLayerParams *params, const DataType dtype){
    LinearLayer linear_layer;

    linear_layer.params  = params;
    linear_layer.dtype = dtype;
    tensor_reset(&linear_layer.output);
    linear_layer_workspace_init(&linear_layer.workspace);
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
    // tensor_print(x, "x");
    // tensor_print(&linear_layer->params->weight, "linear_layer->params->weight");
    tensor_dot_product_(x, &linear_layer->params->weight, &linear_layer->workspace.a);
    //tensor_print(&linear_layer->workspace.a, "linear_layer->workspace.a");
    tensor_add_(&linear_layer->workspace.a, &linear_layer->params->bias, &linear_layer->output);
    //tensor_print(&linear_layer->workspace.output, "linear_layer->workspace.output");
}

// void linear_layer_print(const LinearLayer *layer, const char *heading){
//     printf("\033[33m============================== LINEAR LAYER %s ==============================\033[0m", heading);
//     tensor_print(&layer->weights);
//     tensor_print(&layer->output);
// }
