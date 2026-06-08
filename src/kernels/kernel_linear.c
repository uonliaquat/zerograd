#include "../../inc/kernels/kernel_linear.h"
#include "../../inc/prims/matmul.h"


void kernel_linear_cpu_f32_forward(
    float *weight, float *input, float *bias, 
    float *out,
    size_t m, size_t n, size_t k,
    bool trans_weight
){

    matmul_cpu_f32(input, weight, out, m, k, k, n);
    if(bias != NULL){
        for(size_t i = 0; i < m; i++){
            for(size_t j = 0; j < n; j++){
                out[(i*n) + j] += bias[j];
            }
        }
    }
}