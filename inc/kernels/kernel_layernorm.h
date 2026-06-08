#ifndef __KERNEL_LAYER_NORM__
#define __KERNEL_LAYER_NORM__

#include <stdio.h>


static inline size_t kernel_layernorm_cpu_f32_scratch_bytes(){
    return 0;
}

void kernel_layernorm_cpu_f32_forward(
    const float *embed, const float *weights, const float *bias, 
    float *out, 
    const size_t seq_len, const size_t embed_dim
);
#endif