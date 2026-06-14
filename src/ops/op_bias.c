#include "../../inc/ops/op_bias.h"
#include "../../inc/graph/op_table.h"



Tensor *op_bias(Graph *graph, const char *name, const size_t size, const DType dtype){
    Tensor *out = graph_alloc_node(graph);
    tensor_create(out, name, (size_t[]){size}, 1, NULL, 0, dtype, OP_NONE, NULL, TENSOR_BIAS);
    return out;
}
