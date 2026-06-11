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
    const size_t nheads,
    Tensor *q, Tensor *k, Tensor *v
){
    assert(q->d_type == k->d_type && k->d_type == v->d_type);
    assert(q->ndim == k->ndim && k->ndim == v->ndim);
    assert(q->shape[0] == k->shape[0] && k->shape[0] == v->shape[0]);
    assert(q->shape[1] == k->shape[1] && k->shape[1] == v->shape[1]);
    assert(q->nbytes == k->nbytes && k->nbytes == v->nbytes);

    Tensor *out = graph_alloc_node(graph);

    AttentionParams *op_params = calloc(1, sizeof(AttentionParams));
    op_params->embed_dim = q->shape[1];
    op_params->n_heads = nheads;
    op_params->head_dim = op_params->embed_dim  / nheads;
    op_params->ctx_win = q->shape[0];

    tensor_create(out, name, (size_t[]){op_params->ctx_win, op_params->embed_dim}, 2, (Tensor*[]){q, k, v}, 3, q->d_type, OP_ATTENTION, op_params);
    return out;
}

void op_attention_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));

    AttentionParams * params = (AttentionParams*)tensor->op_params;

    float *q  = &ctx->mem[tensor->src[0]->data_offset];
    float *k  = &ctx->mem[tensor->src[1]->data_offset];
    float *v  = &ctx->mem[tensor->src[2]->data_offset];
    float *out   = &ctx->mem[tensor->data_offset];
    float *scratch  = &ctx->mem[tensor->scratch_offset];
    size_t nbytes_scratch = tensor->nbytes_scratch;

    size_t ctx_win = params->ctx_win;
    size_t embed_dim = params->embed_dim;
    size_t n_heads = params->n_heads;
    size_t head_dim = params->head_dim;


    
    // printf("data_offset:        %zu\n", tensor->data_offset);
    // printf("nbytes:             %zu\n", tensor->nbytes);
    // printf("scratch_offset:     %zu\n", tensor->scratch_offset);
    assert(nbytes_scratch > 0);
    
    

    kernel_multi_head_attention_cpu_f32_forward(q, k, v, out, scratch, ctx_win, embed_dim, n_heads, head_dim);
}