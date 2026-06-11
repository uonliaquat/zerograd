#ifndef __OP_WEIGHT_H__
#define __OP_WEIGHT_H__

// #include "../graph/tensor.h"
#include "../graph/graph.h"

Tensor *op_weight(Graph *graph, const char *name, const size_t rows, const size_t cols, const DType dtype);
    
#endif