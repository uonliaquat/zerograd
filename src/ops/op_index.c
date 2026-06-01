#include "../../inc/ops/op_index.h"
#include "../../inc/kernels/cpu/kernel_index.h"
#include "../../inc/graph/op_table.h"
#include "../../inc/graph/tensor.h"


#include <stdio.h>

void op_index_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));
    
    float *table        = &ctx->mem[tensor->src[0]->data_offset];
    size_t *indices     = &ctx->mem[tensor->src[1]->data_offset];
    float *out          = &ctx->mem[tensor->data_offset];
    size_t seq_len      = tensor->src[0]->shape[0];
    size_t embed_dim    = tensor->src[0]->shape[1];
    size_t size_indices = tensor->src[1]->shape[0];
    size_t size_out     = tensor->nelems;
    
    printf("seq_len: %zu, embed_dim: %zu, size_indices: %zu, size_out: %zu\n", seq_len, embed_dim, size_indices, size_out);
    kernel_index_cpu_f32(table, indices, out, seq_len, embed_dim, size_indices, size_out);
}