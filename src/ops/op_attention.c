#include "../../inc/ops/op_attention.h"
#include "../../inc/kernels/kernel_attention.h"
#include "../../inc/graph/op_table.h"
// #include "../../inc/graph/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


size_t op_attention_scratch_bytes(const Tensor *tensor){
    AttentionParams *params = ((AttentionParams*)(tensor->op_params));
    switch(tensor->d_type){
        case DTYPE_F32: return kernel_multi_head_attention_cpu_f32_scratch_bytes(params->n_heads, params->ctx_win, params->embed_dim);
        default: return 0;
    }
}

Tensor *op_attention(Graph *graph, const char *name, 
    const size_t ctx_win, const size_t ndim, const size_t nheads,
    Tensor *qkv_proj
){
    Tensor *out = graph_alloc_node(graph);

    AttentionParams *op_params = calloc(1, sizeof(AttentionParams));
    op_params->embed_dim = ndim;
    op_params->n_heads = nheads;
    op_params->head_dim = ndim / nheads;

    tensor_create(out, name, (size_t[]){ctx_win, ndim}, 2, (Tensor*[]){qkv_proj}, 1, qkv_proj->d_type, OP_ATTENTION, op_params);
    return out;
}

void op_attention_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));

    float *qkv_proj         = &ctx->mem[tensor->src[0]->data_offset];
    size_t ctx_win          = tensor->src[0]->shape[0];
    size_t qkv_embed_dim    = tensor->src[0]->shape[1];
    size_t embed_dim        = qkv_embed_dim / 3;
    float *query            = qkv_proj;
    float *key              = qkv_proj  + ctx_win;
    float *value            = key + ctx_win;
    float *out              = &ctx->mem[tensor->data_offset];
    float *scratch          = &ctx->mem[tensor->scratch_offset];
    size_t nbytes_scratch   = tensor->nbytes_scratch;

    size_t n_heads = 12;
    size_t head_dim = embed_dim / n_heads;

    
    // printf("data_offset:        %zu\n", tensor->data_offset);
    // printf("nbytes:             %zu\n", tensor->nbytes);
    // printf("scratch_offset:     %zu\n", tensor->scratch_offset);
    assert(nbytes_scratch > 0);
    

    kernel_multi_head_attention_cpu_f32_forward(query, key, value, out, scratch, ctx_win, embed_dim, n_heads, head_dim);
}