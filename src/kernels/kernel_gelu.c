#include "../../inc/kernels/kernel_gelu.h"
#include <math.h>


void kernel_gelu_cpu_f32_forward(float *x, float *out, const size_t n){
    
    const float sqrt_2_over_pi = 0.7978845608f;
    for(size_t i = 0; i < n; i++)
        out[i] = 0.5f * x[i] *
            (1.0f + tanhf(
                sqrt_2_over_pi *
                (x[i] + 0.044715f * x[i] * x[i] * x[i])
            ));

}