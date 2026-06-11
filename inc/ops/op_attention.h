#ifndef __OP_ATTENTION_H__
#define __OP_ATTENTION_H__

// #include "../../inc/graph/context.h"
#include "../../inc/graph/graph.h"
#include <string.h>

typedef struct AttentionParams{
    size_t embed_dim;
    size_t head_dim;
    size_t n_heads;
    size_t ctx_win;
} AttentionParams;

size_t op_attention_scratch_bytes(const Tensor *tensor);

Tensor *op_attention(Graph *graph, const char *name, 
    const size_t ctx_win, const size_t ndim, const size_t nheads,
    Tensor *qkv_proj
);
void op_attention_forward(Context *ctx, Tensor *tensor);

#endif