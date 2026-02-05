#ifndef __LAYER_NORM_H__
#define __LAYER_NORM_H__

#include "./tensor.h"
#include "./linear.h"
#include <stdio.h>

typedef struct LayerNormParams{
    Tensor bias;
    Tensor weight;
} LayerNormParams;

typedef struct LayerNorm{
    LayerNormParams *params;
    double eps;
} LayerNorm;

LayerNorm layer_norm_init(size_t embed_dim);
Tensor layer_norm_forward(LayerNorm *layer_norm, Tensor *x);

#endif