#ifndef __ATTENTION_H__
#define __ATTENTION_H__

#include <stdio.h>


static inline size_t kernel_multi_head_attention_cpu_f32_scratch_bytes(
    const size_t n_heads, const size_t ctx_win, const size_t embed_dim
) {
    size_t head_dim = embed_dim / n_heads;
    size_t per_head =
          ctx_win * head_dim   /* k_t      */
        + ctx_win * ctx_win    /* qk_t     */
        + ctx_win * head_dim;  /* head_out */
    return per_head * sizeof(float);
}

void kernel_multi_head_attention_cpu_f32_forward(
    float *q, float *k, float *v, float *out, float *scratch,
    const size_t ctx_win, const size_t embed_dim, 
    const size_t n_heads, const size_t head_dim
);

#endif