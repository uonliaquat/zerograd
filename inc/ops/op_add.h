#ifndef __OP_ADD_H__
#define __OP_ADD_H__

#include "../../inc/graph/context.h"
#include <string.h>

typedef struct Tensor Tensor;
static inline size_t op_add_scratch_bytes() {
    return 0;
}

void op_add_forward(Context *ctx, Tensor *tensor);

#endif