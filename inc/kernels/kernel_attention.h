#ifndef __ATTENTION_H__
#define __ATTENTION_H__

#include <stdio.h>


static inline size_t kernel_attention_cpu_f32_scratch_bytes(
    const size_t n_heads, const size_t ctx_win, const size_t emebd_dim
) {
    size_t head_dim = emebd_dim / n_heads;
    return (((ctx_win * head_dim) + (ctx_win * ctx_win)) * n_heads) * sizeof(float);
}

void kernel_attention_cpu_f32_forward(
    float *query, float *key, float *value, 
    float *out, float *scratch,
    const size_t ctx_win, const size_t embed_dim, 
    const size_t n_heads, const size_t head_dim
);

#endif