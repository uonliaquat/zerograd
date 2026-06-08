#include "../../inc/ops/op_arange.h"
#include "../../inc/kernels/kernel_arange.h"
#include "../../inc/graph/op_table.h"
// #include "../../inc/graph/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


size_t op_arange_scratch_bytes(const Tensor *tensor){
    switch(tensor->d_type){
        case DTYPE_F32: return kernel_arange_cpu_f32_scratch_bytes();
        default: return 0;
    }
}

Tensor *op_arange(Graph *graph, const char *name, const size_t n){

    Tensor *out = graph_alloc_node(graph);
    tensor_create(out, "token.indices", (size_t[]){n}, 1, NULL, 0, DTYPE_I32, OP_ARANGE, NULL);
    return out;
}
void op_arange_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));
    int *out = &ctx->mem[tensor->data_offset];
    size_t ctx_win = tensor->nelems;

    kernel_arange_cpu_f32_forward(out, ctx_win);
}