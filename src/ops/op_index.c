#include "../../inc/ops/op_index.h"
#include "../../inc/kernels/kernel_index.h"
#include "../../inc/graph/op_table.h"
// #include "../../inc/graph/tensor.h"


#include <stdio.h>
#include <stdlib.h>



size_t op_index_scratch_bytes(const Tensor *tensor){
    switch(tensor->d_type){
        case DTYPE_F32: return kernel_index_cpu_f32_scratch_bytes();
        default: return 0;
    }
}

Tensor *op_index(Graph *graph, const char *name, const size_t vocab_size, const size_t ndim, Tensor *wte, Tensor *token_ids){
    Tensor *out = graph_alloc_node(graph);
    tensor_create(out, name, (size_t[]){vocab_size, ndim}, 2, (Tensor*[]){wte, token_ids}, 2, wte->d_type, OP_INDEX,  NULL, TENSOR_ACTIVATION);
    return out;
}
void op_index_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));

    float *table        = &ctx->mem[tensor->src[0]->data_offset];
    int *indices        = &ctx->mem[tensor->src[1]->data_offset];
    float *out          = &ctx->mem[tensor->data_offset];
    size_t seq_len      = tensor->src[0]->shape[0];
    size_t embed_dim    = tensor->src[0]->shape[1];
    size_t size_indices = tensor->src[1]->nelems;
    size_t size_out     = tensor->nelems;
    
    //printf("seq_len: %zu, embed_dim: %zu, size_indices: %zu, size_out: %zu\n", seq_len, embed_dim, size_indices, size_out);
    kernel_index_cpu_f32_forward(table, indices, out, embed_dim, size_indices);    
    
}