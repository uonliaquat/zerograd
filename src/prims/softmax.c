#include "../../inc/prims/softmax.h"
#include <math.h>

void softmax_cpu_f32(float *a, float *out, size_t size_a){
    float exp_sum = 0;
    for(size_t i = 0; i < size_a; i++){
        exp_sum += expf(a[i]);
    }
    for(size_t i = 0; i < size_a; i++){
        out[i] = expf(a[i]) / exp_sum;
    }
}