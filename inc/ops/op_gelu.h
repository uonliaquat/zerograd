#ifndef __OP_GELU_H__
#define __OP_GELU_H__

#include "../../inc/graph/context.h"
#include <string.h>

typedef struct Tensor Tensor;
static inline size_t op_gelu_scratch_bytes() {
    return 0;
}

void op_gelu_forward(Context *ctx, Tensor *tensor);

#endif