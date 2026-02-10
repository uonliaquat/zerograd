#ifndef __LINEAR_LAYER_H__
#define __LINEAR_LAYER_H__

#include <stdio.h>
#include "../tensor.h"


typedef struct LinearLayerWorkspace{
    Tensor a;
    Tensor output;
} LinearLayerWorkspace;


typedef struct LinearLayerParams{
    Tensor bias;
    Tensor weight;
} LinearLayerParams;

typedef struct LinearLayer{
    char name[128];
    LinearLayerParams *params;
    LinearLayerWorkspace workspace;
}LinearLayer;


LinearLayer linear_layer_init(LinearLayerParams *params, char *name);
void        linear_layer_free(const LinearLayer *layer);
void        linear_layer_forward(LinearLayer *layer, Tensor *x);
void        linear_layer_write(LinearLayer *layer, Tensor **tensors, size_t *tensors_len);

#endif