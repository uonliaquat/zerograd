#include "../../inc/kernels/kernel_add.h"
#include <assert.h>

void kernel_add_cpu_f32_forward(
    const float *a, const float *b, 
    float *out, 
    const size_t n
){
    for(size_t i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
} 
