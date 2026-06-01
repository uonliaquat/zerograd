#include "../../../inc/kernels/cpu/kernel_add.h"
#include <assert.h>

void kernel_add_cpu_f32(
    const float *a, const float *b, float *out, 
    const size_t size_a, const size_t size_b, const size_t size_out
){
    assert(size_a == size_b && size_b == size_out);
    for(size_t i = 0; i < size_a; i++){
        out[i] = a[i] + b[i];
    }
}