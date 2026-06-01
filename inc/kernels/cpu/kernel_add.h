#ifndef __KERNEL_ADD_H__
#define __KERNEL_ADD_H__

#include <string.h>

void kernel_add_cpu_f32(
    const float *a, const float *b, float *c, 
    const size_t size_a, const size_t size_b, const size_t size_c
);

#endif