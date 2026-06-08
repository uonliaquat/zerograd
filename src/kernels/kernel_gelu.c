#include "../../inc/kernels/kernel_gelu.h"
#include <math.h>


void kernel_gelu_cpu_f32_forward(const float *x, float *out){
    const float sqrt_2_over_pi = 0.7978845608f;

    *out = 0.5f * (*x) *
           (1.0f + tanhf(
               sqrt_2_over_pi *
               ((*x) + 0.044715f * (*x) * (*x) * (*x))
           ));

}