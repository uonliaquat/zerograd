#ifndef __OP_LAYER_NORM_H__
#define __OP_LAYER_NORM_H__

#include <string.h>

typedef struct Tensor Tensor;
static inline size_t op_layernorm_scratch_bytes(){
    return 0;
}

void op_layernorm_forward(Tensor *tensor);

#endif