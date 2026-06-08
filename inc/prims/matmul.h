#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <stdio.h>
#include <stdbool.h>

void matmul_cpu_f32(float *mat1, float *mat2, float *out, 
    size_t rows_mat1, size_t cols_mat1, size_t rows_mat2, size_t cols_mat2,
    bool trans_weight
);

#endif