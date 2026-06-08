#ifndef __OP_INDEX_H__
#define __OP_INDEX_H__

// #include "../../inc/graph/context.h"
#include "../../inc/graph/graph.h"
#include <string.h>

// typedef struct Tensor Tensor;
// typedef struct Graph Graph;

size_t op_index_scratch_bytes(const Tensor *tensor);
Tensor *op_index(Graph *graph, const char *name, const size_t vocab_size, const size_t ndim, Tensor *wte, Tensor *token_ids);
void op_index_forward(Context *ctx, Tensor *tensor);

#endif