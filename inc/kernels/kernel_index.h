#ifndef __KERNEL_INDEX_H__
#define __KERNEL_INDEX_H__

#include <stdio.h>


static inline size_t kernel_index_cpu_f32_sctach_bytes() {
    return 0;
}

void kernel_index_cpu_f32_forward(
    const float *table, const int *indices, 
    float *out,
    const size_t emebd_dim, const size_t size_indices
);

#endif