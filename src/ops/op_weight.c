#include "../../inc/ops/op_weight.h"
#include "../../inc/graph/op_table.h"



Tensor *op_weight(Graph *graph, const char *name, const size_t rows, const size_t cols, const DType dtype){
    Tensor *out = graph_alloc_node(graph);
    tensor_create(out, name, (size_t[]){rows, cols}, 2, NULL, 0, dtype, OP_NONE, NULL);
    return out;
}
