#include "../../inc/kernels/kernel_layernorm.h"

#include <math.h>
#include <stdlib.h>

void kernel_layernorm_cpu_f32_forward(
    const float *embed, const float *weights, const float *bias, 
    float *out, 
    const size_t seq_len, const size_t embed_dim,
    const float eps
){
    for(size_t i = 0; i < seq_len; i++){
        double mean = 0;
        double variance = 0;
        for(size_t j = 0; j < embed_dim; j++){
            mean += embed[(i*embed_dim) + j];
        }
        mean = mean / embed_dim;

        for(size_t j = 0; j < embed_dim; j++){
            float diff = embed[(i*embed_dim) + j] - mean;
            variance += (diff * diff);
        }
        variance = variance / embed_dim;

        for(size_t j = 0; j < embed_dim; j++){
            float norm = (embed[(i*embed_dim) + j] - mean) / sqrt(variance + eps);
            out[(i*embed_dim)+j] = norm * weights[j] + bias[j];
        }
    }
}