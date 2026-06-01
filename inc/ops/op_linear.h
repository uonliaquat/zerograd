#ifndef __OP_LINEAR_H__
#define __OP_LINEAR_H__

#include <string.h>

typedef struct Tensor Tensor;
static inline size_t op_linear_scratch_bytes(){
    return 0;
}

void op_linear_forward(Tensor *tensor);

#endif