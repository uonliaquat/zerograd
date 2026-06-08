#ifndef __GELU_H__
#define __GELU_H__

#include <stdio.h>

static inline size_t kernel_gelu_cpu_f32_sctach_bytes() {
    return 0;
}

void kernel_gelu_cpu_f32_forward(const float *x, float *out);
#endif