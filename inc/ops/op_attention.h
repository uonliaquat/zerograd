#ifndef __OP_ATTENTION_H__
#define __OP_ATTENTION_H__

#include "../../inc/graph/context.h"
#include <string.h>

typedef struct Tensor Tensor;
static inline size_t op_attention_scratch_bytes() {
    return 0;
}

void op_attention_forward(Context *ctx, Tensor *tensor);

#endif