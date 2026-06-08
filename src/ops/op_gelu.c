#include "../../inc/ops/op_gelu.h"
#include "../../inc/kernels/kernel_gelu.h"
#include "../../inc/graph/op_table.h"
// #include "../../inc/graph/tensor.h"

#include <stdio.h>
#include <stdlib.h>

size_t op_gelu_scratch_bytes(const Tensor *tensor){
    switch(tensor->d_type){
        case DTYPE_F32: return kernel_gelu_cpu_f32_scratch_bytes();
        default: return 0;
    }
}

void *op_gelu(Graph *graph, const char *name, const size_t ctx_win, const size_t ndim,
            Tensor *src1)
{
    Tensor *out = graph_alloc_node(graph);
    tensor_create(out, name, (size_t[]){ctx_win, ndim}, 2, (Tensor*[]){src1}, 1, src1->d_type, OP_GELU, NULL);
    return out;
}
void op_gelu_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));

    float *mlp_proj = &ctx->mem[tensor->src[0]->data_offset];
    size_t elems    = tensor->nelems;
    float *out      = &ctx->mem[tensor->data_offset];

    for(size_t i = 0 ; i < elems; i++){
        kernel_gelu_cpu_f32_forward(mlp_proj + i, out + i);
    }
}