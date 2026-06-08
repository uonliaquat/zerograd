#ifndef __OP_ADD_H__
#define __OP_ADD_H__

// #include "../../inc/graph/context.h"
// #include "../../inc/graph/tensor.h"
#include "../../inc/graph/graph.h"
#include <string.h>

// typedef struct Tensor Tensor;

size_t op_add_scratch_bytes(const Tensor *tensor);
Tensor *op_add(Graph *graph, const char *name, Tensor *src1, Tensor *src2);
void op_add_forward(Context *ctx, Tensor *tensor);

#endif