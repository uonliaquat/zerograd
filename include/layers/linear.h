#ifndef __LINEAR_LAYER_H__
#define __LINEAR_LAYER_H__

#include <stdio.h>
#include <stdbool.h>
#include "../tensor.h"


typedef struct LinearLayerWorkspace{
    Tensor a;
} LinearLayerWorkspace;

typedef struct LinearLayerParams{
    Tensor bias;
    Tensor weight;
} LinearLayerParams;

typedef struct LinearLayer{
    LinearLayerParams *params;
    DataType dtype;
    char name[128];
    LinearLayerWorkspace workspace;
    Tensor output;
}LinearLayer;

// LinearLayer linear_layer_init(const size_t inputs, const size_t outputs, const bool bias, const bool requires_grad, const DataType dtype);
LinearLayer linear_layer_init(LinearLayerParams *params, const DataType dtype, char *name);
void        linear_layer_params_free(const LinearLayerParams *params);
void        linear_layer_free(const LinearLayer *linear_layer);
void        linear_layer_forward(LinearLayer *linear_layer, Tensor *x);
void        linear_layer_write(LinearLayer *linear_layer, Tensor **tensors, size_t *tensors_len);
//void        linear_layer_print(const LinearLayer *layer, const char *heading);

#endif