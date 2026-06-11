#ifndef __OP_BIAS_H__
#define __OP_BIAS_H__

// #include "../graph/tensor.h"
#include "../graph/graph.h"

Tensor *op_bias(Graph *graph, const char *name, const size_t size, const DType dtype);
    
#endif