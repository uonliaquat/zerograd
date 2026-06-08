#ifndef __OP_LINEAR_H__
#define __OP_LINEAR_H__

// #include "../../inc/graph/context.h"
#include "../../inc/graph/graph.h"
#include <string.h>

// typedef struct Tensor Tensor;
// typedef struct Graph Graph;

size_t op_linear_scratch_bytes(const Tensor *tensor);
Tensor *op_linear(
    Graph *graph, const char *name,
    Tensor *weight, Tensor *bias, Tensor *input,
    bool trans_weight
);
void op_linear_forward(Context *ctx, Tensor *tensor);

#endif