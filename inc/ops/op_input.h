#ifndef __OP_INPUT_H__
#define __OP_INPUT_H__

// #include "../graph/tensor.h"
#include "../graph/graph.h"

// typedef struct Graph Graph;

Tensor *op_input(Graph *graph, const char *name, const size_t n);

#endif