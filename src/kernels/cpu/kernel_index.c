#include "../../../inc/kernels/cpu/kernel_index.h"

#include <string.h>

void kernel_index_cpu_f32(
    const float *table, const size_t *indices, float *out,
    const size_t seq_len, const size_t embed_dim, const size_t size_indices, const size_t size_out
){
    for(size_t i = 0; i < size_indices; i++){
        size_t token_id = indices[i];
        token_id = 1; //this needs to be removed in future
        memcpy(out, &table[token_id * embed_dim], embed_dim * sizeof(float));
        out += embed_dim * sizeof(float);
    }
}