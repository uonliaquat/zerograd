#ifndef __KERNEL_ADD_H__
#define __KERNEL_ADD_H__

#include <stdio.h>


static inline size_t kernel_add_cpu_f32_scratch_bytes() {
    return 0;
}


void kernel_add_cpu_f32_forward(
    const float *a, const float *b, 
    float *out, 
    const size_t n
);

#endif