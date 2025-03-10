#include "../include/linalg.h"

void gemm(Tensor *a, Tensor *b, Tensor *c){
    clock_t start = clock();
    for(size_t row_c = 0; row_c < c->rows; row_c++){
        for(size_t col_c = 0; col_c < c->cols; col_c++){
            double val_c = 0;
            for(size_t col_a = 0; col_a < a->cols; col_a++){
                float val_a = GET_ELEMENT(a, row_c, col_a);
                float val_b = GET_ELEMENT(b, col_a, col_c);
                val_c += (val_a * val_b);
            }
            PUT_ELEMENT(c, row_c, col_c, val_c);
        }
    }
    clock_t end = clock();
    double time_seconds = (double)(end - start)/CLOCKS_PER_SEC;
    double flop = 2*(c->rows * c->cols * a->cols);
    double gflops = (flop / time_seconds) * 1e-9;
    printf("Time(s): %.4f \n", time_seconds);
    printf("GFLOPS:  %.4f \n", gflops);
}