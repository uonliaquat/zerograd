#ifndef __LAYER_NORM_H__
#define __LAYER_NORM_H__

#include "./tensor.h"
#include "./linear.h"
#include <stdio.h>

typedef struct LayerNormWorkSpace{
    Tensor mean_var;
    Tensor x_norm;
    Tensor x_norm_scaled;
    Tensor x_norm_shifted;
    Tensor output;
} LayerNormWorkSpace;


typedef struct LayerNormParams{
    Tensor bias;
    Tensor weight;
} LayerNormParams;

typedef struct LayerNorm{
    float eps;
    char name[128];
    LayerNormParams *params;
    LayerNormWorkSpace workspace;
} LayerNorm;

LayerNorm   layer_norm_init(LayerNormParams *params, char *name);
void        layer_norm_free(LayerNorm *layer_norm);
void        layer_norm_forward(LayerNorm *layer_norm, Tensor *x);
void        layer_norm_write(LayerNorm *layer_norm, Tensor **tensors, size_t *tensors_len);

#endif