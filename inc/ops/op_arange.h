#ifndef __OP_ARANGE__
#define __OP_ARANGE__


#include "../../inc/graph/graph.h"
#include <string.h>


// typedef struct Tensor Tensor;

size_t op_arange_scratch_bytes(const Tensor *tensor);
Tensor *op_arange(Graph *graph, const char *name, const size_t n);
void op_arange_forward(Context *ctx, Tensor *tensor);

#endif