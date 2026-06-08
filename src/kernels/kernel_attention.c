#include "../../inc/kernels/kernel_attention.h"
#include "../../inc/prims/matmul.h"
#include "../../inc/prims/softmax.h"

#include <math.h>
#include <stdlib.h>

void kernel_attention_cpu_f32_forward(
    float *query, float *key, float *value, 
    float *out, float *scratch,
    const size_t ctx_win, const size_t embed_dim, 
    const size_t n_heads, const size_t head_dim
){
        
        for(size_t head = 0; head < n_heads; head++){
            size_t offset = (head * ctx_win * head_dim);
            float *q = query + offset;
            float *k = key + offset;
            float *v = value + offset;
            float * head_out = out;
            float *k_t = scratch + head * ((ctx_win * head_dim) + (ctx_win * ctx_win));
            //key transpose
            for(size_t i = 0; i < ctx_win; i++){
                for(size_t j = 0; j < head_dim; j++){
                    float val = k[(i * head_dim) + j];
                    k_t[(j * ctx_win) + i] = val;
                }
            }

            for(size_t i = 0; i < 10; i++){
                printf("%.3f, ", k[i]);
            }
            exit(1);


            //Q.K^t
            float *qk_t = k_t + (ctx_win * head_dim);
            matmul_cpu_f32(q, k_t, qk_t, ctx_win, head_dim, head_dim, ctx_win);

            // printf("\n\nAttention Scores\n");
            // for(size_t i = 0; i < 10; i++){
            //     printf("%.3f, ", qk_t[i]);
            // }

            // // (Q.K^t) / sqrt(head_dim)
            for(size_t i = 0; i < ctx_win * ctx_win; i++){
                qk_t[i] = qk_t[i] / sqrt(head_dim);
            }
            // printf("\n\nScaled Attention Scores\n");
            // for(size_t i = 0; i < 10; i++){
            //     printf("%.3f, ", qk_t[i]);
            // }


            // //causal mask
            for(size_t i = 1; i < ctx_win; i++){
                for(size_t j = i; j < ctx_win; j++){
                    qk_t[((i-1) * ctx_win) + j] = -INFINITY;
                }
            }

            // printf("\n\nScaled Causal Attention Scores\n");
            // for(size_t i = 0; i < 10; i++){
            //     printf("%.3f, ", qk_t[i]);
            // }

            // // softmax((Q.K^t) / sqrt(head_dim))
            for(size_t i = 0; i < ctx_win; i++){
                softmax_cpu_f32(qk_t+(i*ctx_win), qk_t+(i*ctx_win), ctx_win);
            }

            // printf("\n\nAttention Weights\n");
            // for(size_t i = 0; i < 10; i++){
            //     printf("%.3f, ", qk_t[i]);
            // }

  

            // softmax((Q.K^t) / sqrt(head_dim)) * V
            matmul_cpu_f32(qk_t, v, head_out, ctx_win, ctx_win, ctx_win, head_dim);

            // printf("\n\nkast\n");
            // for(size_t i = 0; i < 10; i++){
            //     printf("%.3f, ", head_out[i]);
            // }
            // printf("\n");
            //exit(1);


        }

}