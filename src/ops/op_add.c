#include "../../inc/ops/op_add.h"
#include "../../inc/graph/op_table.h"
// #include "../../inc/graph/tensor.h"
#include "../../inc/kernels/kernel_add.h"

#include <stdio.h>
#include <assert.h>


size_t op_add_scratch_bytes(const Tensor *tensor){
    switch(tensor->d_type){
        case DTYPE_F32: return kernel_add_cpu_f32_sctach_bytes();
        default: return 0;
    }
}

Tensor *op_add(Graph *graph, const char *name, Tensor *src1, Tensor *src2){
    assert(src1->d_type == src2->d_type);
    assert(src1->ndim == src2->ndim && src1->ndim == 2);
    assert(src1->shape[0] == src2->shape[0]);
    assert(src1->shape[1] == src2->shape[1]);
    size_t rows = src1->shape[0];
    size_t cols = src2->shape[1];
    Tensor *out = graph_alloc_node(graph);
    
    tensor_create(out, name, (size_t[]){rows, cols}, 2, (Tensor*[]){src1, src2}, 2, src1->d_type, OP_ADD, NULL);
    return out;
}

void op_add_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));
    float *a            = &ctx->mem[tensor->src[0]->data_offset];
    float *b            = &ctx->mem[tensor->src[1]->data_offset];
    float *out          = &ctx->mem[tensor->data_offset];
    size_t size_a       = tensor->src[0]->nelems;
    size_t size_b       = tensor->src[1]->nelems;
    size_t size_out     = tensor->nelems;
    
    assert(size_a == size_b && size_b == size_out);
    kernel_add_cpu_f32_forward(a, b, out, size_a);
}