#include "../../inc/kernels/kernel_attention.h"
// #include "../../inc/prims/matmul.h"
#include "../../inc/prims/softmax.h"

#include <math.h>
#include <stdlib.h>

/* C = A · B, where A and B are strided views; out is packed [rows_mat1, cols_mat2].
 *   mat1 : [rows_mat1, cols_mat1]    mat2 : [rows_mat2, cols_mat2]
 *   requires cols_mat1 == rows_mat2 (inner dimensions must match)
 *   stride_row_* : elements to advance one row;  stride_col_* : one column
 *   out must not alias mat1 or mat2.
 */
void matmul_strided_cpu_f32(const float *mat1, const float *mat2, float *out,
    size_t rows_mat1, size_t cols_mat1, size_t rows_mat2, size_t cols_mat2,
    size_t row_stride_mat1, size_t col_stride_mat1,
    size_t row_stride_mat2, size_t col_stride_mat2
){
    (void)rows_mat2;   /* must equal cols_mat1; kept for signature symmetry */

    for (size_t i = 0; i < rows_mat1; i++) {
        for (size_t j = 0; j < cols_mat2; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_mat1; k++) {
                float a = mat1[i * row_stride_mat1 + k * col_stride_mat1];
                float b = mat2[k * row_stride_mat2 + j * col_stride_mat2];
                sum += a * b;
            }
            out[i * cols_mat2 + j] = sum;
        }
    }
}

void transpose_strided_cpu_f32(const float *in, float *out,
    size_t in_rows, size_t in_cols,
    size_t stride_row, size_t stride_col
){
    for (size_t i = 0; i < in_rows; i++) {
        for (size_t j = 0; j < in_cols; j++) {
            out[j * in_rows + i] = in[i * stride_row + j * stride_col];
        }
    }
}

void kernel_multi_head_attention_cpu_f32_forward(
    float *q, float *k, float *v, float *out, float *scratch,
    const size_t ctx_win, const size_t embed_dim,
    const size_t n_heads, const size_t head_dim
){
    for (size_t head = 0; head < n_heads; head++) {

        size_t off = head * head_dim;   /* this head's column offset into q/k/v/out */

        /* --- key transpose: K slice [ctx_win, head_dim] -> k_t [head_dim, ctx_win] packed --- */
        float *k_t = scratch;
        transpose_strided_cpu_f32(k + off, k_t, ctx_win, head_dim, embed_dim, 1);

        /* --- Q · Kᵀ ---
           Q : view, strides (embed_dim, 1)   k_t : packed [head_dim, ctx_win], strides (ctx_win, 1) */
        float *qk_t = &k_t[head_dim * ctx_win];
        matmul_strided_cpu_f32(
            q + off, k_t, qk_t,
            ctx_win, head_dim,        /* A [ctx_win, head_dim]  */
            head_dim, ctx_win,        /* B [head_dim, ctx_win]  */
            embed_dim, 1,             /* A strides: row = embed_dim */
            ctx_win, 1                /* B strides: packed          */
        );

        /* --- scale --- */
        float inv = 1.0f / sqrtf((float)head_dim);
        for (size_t i = 0; i < ctx_win * ctx_win; i++)
            qk_t[i] *= inv;

        /* --- causal mask: query row r masks key cols c > r --- */
        for (size_t row = 0; row < ctx_win; row++)
            for (size_t col = row + 1; col < ctx_win; col++)
                qk_t[row * ctx_win + col] = -INFINITY;

        /* --- row-wise softmax --- */
        for (size_t row = 0; row < ctx_win; row++)
            softmax_cpu_f32(qk_t + row * ctx_win, qk_t + row * ctx_win, ctx_win);

        /* --- scores · V ---
           scores : packed [ctx_win, ctx_win], strides (ctx_win, 1)
           V      : view  [ctx_win, head_dim], strides (embed_dim, 1)
           => head_out [ctx_win, head_dim] packed */
        float *head_out = &qk_t[ctx_win * ctx_win];
        matmul_strided_cpu_f32(
            qk_t, v + off, head_out,
            ctx_win, ctx_win,         /* A [ctx_win, ctx_win]  */
            ctx_win, head_dim,        /* B [ctx_win, head_dim] */
            ctx_win, 1,               /* A strides: packed         */
            embed_dim, 1              /* B strides: V row = embed_dim */
        );

        /* --- scatter head_out back into out [ctx_win, embed_dim] this head's columns --- */
        for (size_t row = 0; row < ctx_win; row++)
            for (size_t col = 0; col < head_dim; col++)
                out[row * embed_dim + off + col] = head_out[row * head_dim + col];
    }
}

// void kernel_multi_head_attention_cpu_f32_forward(
//     float *q, float *k, float *v, float *out, float *scratch,
//     const size_t ctx_win, const size_t embed_dim, 
//     const size_t n_heads, const size_t head_dim
// ){  

//         // printf("head_dim: %zu\n", head_dim);
//         // printf("ctx_win * embed_dim: %zu\n", ctx_win * embed_dim);
        
//         for(size_t head = 0; head < n_heads; head++){

//             // //reorder query
//             float *q_reordered = scratch;
//             size_t scratch_index = 0;
//             for(size_t i = (head*head_dim); i < (ctx_win * embed_dim); i += embed_dim){
//                 for(size_t j = 0; j < head_dim; j++){
//                     // printf("%zu\n", i+j);
//                     q_reordered[scratch_index++]  = q[i+j];
//                 }
//             }
//             // printf("\nq\n");
//             // for(size_t i = 0; i < 10; i++){
//             //     printf("%.4f, ", q_reordered[i]);
//             // }
//             // printf("\ndone\n");

//             // printf("\n\nk\n");
//             // for(size_t i = 0; i < 10; i++){
//             //     printf("%.4f, ", k[i]);
//             // }
 
//             // printf("\n\nv\n");
//             // for(size_t i = 0; i < 10; i++){
//             //     printf("%.4f, ", v[i]);
//             // }
//             // printf("\n\n");



       


//             //key transpose
//             float *k_t = &q_reordered[scratch_index];
//             scratch_index = 0;
//             for(size_t i = 0; i < head_dim; i++){
//                 for(size_t j = (head*head_dim); j < (ctx_win * embed_dim); j += embed_dim){
//                     float val = k[i + j];
//                     k_t[scratch_index++] = val;
//                     // printf("%zu\n", i+j);
//                 }
//             }


//             // printf("\n\nK^T\n");
//             // for(size_t i = 0; i < 10; i++){
//             //     printf("%.4f, ", k_t[i]);
//             // }

//             // printf("\n\n");

       

//             // //Q.K^t
//             float *qk_t = &k_t[scratch_index];
//             matmul_cpu_f32(q_reordered, k_t, qk_t, ctx_win, head_dim, head_dim, ctx_win, true);



//             // printf("\n\nAttention Scores\n");
//             // for(size_t i = 0; i < 10; i++){
//             //     printf("%.3f, ", qk_t[i]);
//             // }



//             // (Q.K^t) / sqrt(head_dim)
//             for(size_t i = 0; i < ctx_win * ctx_win; i++){
//                 qk_t[i] = qk_t[i] / sqrt(head_dim);
//             }

//             // printf("\n\nScaled \n");
//             // for(size_t i = 0; i < 10; i++){
//             //     printf("%.3f, ", qk_t[i]);
//             // }

    

    

//             // causal mask
//             for(size_t i = 1; i < ctx_win; i++){
//                 for(size_t j = i; j < ctx_win; j++){
//                     qk_t[((i-1) * ctx_win) + j] = -INFINITY;
//                 }
//             }


//             // printf("\n\nMasked\n");
//             // for(size_t i = 0; i < 10; i++){
//             //     printf("%.3f, ", qk_t[i]);
//             // }

        

//             // softmax((Q.K^t) / sqrt(head_dim))
//             for(size_t i = 0; i < ctx_win; i++){
//                 softmax_cpu_f32(qk_t+(i*ctx_win), qk_t+(i*ctx_win), ctx_win);
//             }


//             // printf("\n\nSoftmax\n");
//             // for(size_t i = 0; i < 10; i++){
//             //     printf("%.3f, ", qk_t[i]);
//             // }


//             // value reordered
//             float *v_reordered = &qk_t[ctx_win*ctx_win];
//             scratch_index = 0;
//             for(size_t i = (head*head_dim); i < (ctx_win * embed_dim); i += embed_dim){
//                 for(size_t j = 0; j < head_dim; j++){
//                     // printf("%zu\n", i+j);
//                     v_reordered[scratch_index++]  = v[i+j];
//                 }
//             }

//             // // softmax((Q.K^t) / sqrt(head_dim)) * V
//             float * head_out = &v_reordered[scratch_index];
//             matmul_cpu_f32(qk_t, v_reordered, head_out, ctx_win, ctx_win, ctx_win, head_dim, true);




//             float * head_out_reordered = out + (ctx_win*head_dim) * head;
//             scratch_index = 0;
//             for(size_t i = (head*head_dim); i < (ctx_win * embed_dim); i += embed_dim){
//                 for(size_t j = 0; j < head_dim; j++){
//                     // printf("%zu\n", i+j);
//                     head_out_reordered[i+j] = head_out[scratch_index++];
//                 }
//             }
     


  


//             // printf("\n\ny\n");
//             // for(size_t i = 0; i < 10; i++){
//             //     printf("%.3f, ", head_out[i]);
//             // }
//             // printf("\n\n");

//         }

// }