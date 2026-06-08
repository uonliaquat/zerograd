#ifndef __KERNEL_LINEAR_H__
#define __KERNEL_LINEAR_H__

#include <stdio.h>
#include <stdbool.h>


static inline size_t kernel_linear_cpu_f32_scratch_bytes(){
    return 0;
}


void kernel_linear_cpu_f32_forward(
    float *weight, float *input, float *bias, 
    float *out,
    size_t m, size_t n, size_t k,
    bool trans_weight
);

#endif