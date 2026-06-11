#ifndef __OP_LAYER_NORM_H__
#define __OP_LAYER_NORM_H__

// #include "../../inc/graph/context.h"
#include "../../inc/graph/graph.h"
#include <string.h>

typedef struct LayerNormParams {
    float eps;
} LayerNormParams;

size_t op_layernorm_scratch_bytes(const Tensor *tensor);
Tensor *op_layernorm(Graph *graph, const char *name,
    Tensor *weight, Tensor *bias, Tensor *input);
void op_layernorm_forward(Context *ctx, Tensor *tensor);

#endif