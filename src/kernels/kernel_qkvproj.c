#include "../../inc/kernels/kernel_qkvproj.h"
#include "../../inc/prims/matmul.h"


void kernel_qkv_proj_cpu_f32_forward(
    float *input, float *weight, float *bias, 
    float *out,
    size_t ctx_win, size_t embed_dim, size_t n_heads, size_t head_dim
){

    size_t qkv_dim = embed_dim * 3;
    for(size_t i = 0; i < ctx_win; i++){
        for(size_t j = 0; j < qkv_dim; j++){
            float sum = 0;
            for(size_t k = 0; k < embed_dim; k++){
                sum += input[(i*embed_dim) +k] * weight[(k * qkv_dim) + j];
            }
            out[j] = sum;
        }
    }
}