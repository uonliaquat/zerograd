#ifndef __KERNEL_ARANGE_H__
#define __KERNEL_ARANGE_H__

#include <stdio.h>


static inline size_t kernel_arange_cpu_f32_scratch_bytes() {
    return 0;
}

void kernel_arange_cpu_f32_forward(int *out, size_t n);

#endif