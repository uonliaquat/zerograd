#include "../../inc/kernels/kernel_arange.h"

#include <math.h>
#include <stdlib.h>

void kernel_arange_cpu_f32_forward(int *out, size_t n){
    for(size_t i = 0; i < n; i++){
        out[i] = i;
    }
}