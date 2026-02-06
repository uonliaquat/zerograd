#ifndef __LAYER_NORM_H__
#define __LAYER_NORM_H__

#include "./tensor.h"
#include "./linear.h"
#include <stdio.h>

typedef struct LayerNormParams{
    Tensor bias;
    Tensor weight;
} LayerNormParams;

typedef struct LayerNormWorkSpace{
    Tensor mean_var;
    Tensor x_norm;
    Tensor x_norm_scaled;
} LayerNormWorkSpace;

typedef struct LayerNorm{
    LayerNormParams *params;
    float eps;
    LayerNormWorkSpace workspace;
    Tensor output;
} LayerNorm;

LayerNorm   layer_norm_init(LayerNormParams *params, const DataType dtype);
void        layer_norm_free(LayerNorm *layer_norm);
void        layer_norm_forward(LayerNorm *layer_norm, Tensor *x);

#endif