#ifndef __QKV_PROJ_H__
#define __QKV_PROJ_H__


#include "../../inc/graph/graph.h"
#include <stdio.h>



size_t op_qkv_proj_scratch_bytes(const Tensor *tensor);

Tensor *op_qkv_proj(
    Graph *graph, const char *name,
    Tensor *weiight, Tensor *bias, Tensor *input
);
void op_qkv_proj_forward(Context *ctx, Tensor *tensor);

#endif