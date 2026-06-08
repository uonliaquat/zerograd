#include "../../inc/prims/matmul.h"


void matmul_cpu_f32(float *mat1, float *mat2, float *out, 
    size_t rows_mat1, size_t cols_mat1, size_t rows_mat2, size_t cols_mat2){

    for(size_t i = 0; i < rows_mat1; i++){
        for(size_t j = 0; j < cols_mat2; j++){
            float sum = 0;
            for(size_t k = 0; k < cols_mat1; k++){
                sum  += mat1[((i * cols_mat1) + k)] * mat2[(k * cols_mat2) + j];
            }
            out[(i * cols_mat2) + j] = sum;
        }
    }

}