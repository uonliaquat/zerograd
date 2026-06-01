#ifndef __KERNEL_INDEX_H__
#define __KERNEL_INDEX_H__

#include <string.h>

void kernel_index_cpu_f32(
    const float *table, const size_t *indices, float *out,
    const size_t seq_len, const size_t emebd_dim, const size_t size_indices, const size_t size_out
);

#endif