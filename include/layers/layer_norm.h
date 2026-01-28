#ifndef __LAYER_NORM_H__
#define __LAYER_NORM_H__

#include "./tensor.h"
#include <stdio.h>

typedef struct LayerNorm{
    Tensor scale;
    Tensor shift;
    double eps;
} LayerNorm;

LayerNorm layer_norm_init(size_t embed_dim);
Tensor layer_norm_forward(LayerNorm *layer_norm, Tensor *x);

#endif