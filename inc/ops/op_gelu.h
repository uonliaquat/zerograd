#ifndef __OP_GELU_H__
#define __OP_GELU_H__

// #include "../../inc/graph/context.h"
#include "../../inc/graph/graph.h"
#include <string.h>

// typedef struct Tensor Tensor;
// typedef struct Graph Graph;

size_t op_gelu_scratch_bytes(const Tensor *tensor);
void *op_gelu(Graph *graph, const char *name, const size_t ctx_win, const size_t ndim,
            Tensor *src1);
void op_gelu_forward(Context *ctx, Tensor *tensor);

#endif