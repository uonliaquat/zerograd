#ifndef __KERNEL_QKV_PROJ_H__
#define __KERNEL_QKV_PROJ_H__


#include <stdio.h>

static inline size_t kernel_qkv_proj_cpu_f32_scratch_bytes(){
    return 0;
}


void kernel_qkv_proj_cpu_f32_forward(
    float *input, float *weight, float *bias, 
    float *out,
    size_t ctx_win, size_t embed_dim, size_t n_heads, size_t head_dim
);

#endif