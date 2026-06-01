#ifndef __OP_INDEX_H__
#define __OP_INDEX_H__

#include <string.h>

typedef struct Tensor Tensor;

static inline size_t op_index_scratch_bytes() {
    return 0;
}

void op_index_forward(Tensor *tensor);

#endif