#include "../../inc/kernels/kernel_index.h"

#include <string.h>
#include <stdlib.h>

void kernel_index_cpu_f32_forward(
    const float *table, const int *indices, 
    float *out,
    const size_t embed_dim, const size_t size_indices
){
    for(size_t i = 0; i < size_indices; i++){
        int id = indices[i];
        memcpy(out, &table[id * embed_dim], embed_dim * sizeof(int));
        out += embed_dim;
    }
}