#include "../../inc/ops/op_add.h"
#include "../../inc/graph/op_table.h"
#include "../../inc/graph/tensor.h"
#include "../../inc/kernels/cpu/kernel_add.h"
#include <stdio.h>

void op_add_forward(Context *ctx, Tensor *tensor){
    //printf("Executing %s\n", op_name(tensor->op_type));
    float *a            = &ctx->mem[tensor->src[0]->data_offset];
    float *b            = &ctx->mem[tensor->src[1]->data_offset];
    float *out          = &ctx->mem[tensor->data_offset];
    size_t size_a       = tensor->src[0]->nelems;
    size_t size_b       = tensor->src[1]->nelems;
    size_t size_out     = tensor->nelems;

    kernel_add_cpu_f32(a, b, out, size_a, size_b, size_out);
}