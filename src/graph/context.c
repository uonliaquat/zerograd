#include "../../inc/graph/context.h"
#include <assert.h>
#include <stdlib.h>

Context context_init(size_t nbytes){
    Context ctx;
    ctx.size = 0;
    ctx.mem = calloc(nbytes, 1);
    ctx.nbytes = nbytes;
    return ctx;
}

void context_free(Context *ctx){
    ctx->size = 0;
    ctx->nbytes = 0;
    free(ctx->mem);
}

void context_reset(Context *ctx){
    ctx->size = 0;
}

// size_t context_alloc(size_t nbytes){
//     return 0;
// }

// Tensor *context_get_node(Context *ctx){
//     assert(ctx->nnodes < GRAPH_MAX_NODES);
//     Tensor *out = &ctx->nodes[ctx->nnodes];
//     ctx->nnodes++;
//     return out;
// }
