
#include "../../inc/ops/op_input.h"
#include "../../inc/graph/op_table.h"

Tensor *op_input(Graph *graph, const char *name, const size_t n){
    Tensor *out = graph_alloc_node(graph);
    tensor_create(out, name, (size_t[]){n}, 1, NULL, 0, DTYPE_I32, OP_NONE, NULL);
    return out;
}