// dotprod.c
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>   // OpenBLAS BLAS interface

int main() {
    //Taking dot product

    int m = 2, k = 3, n = 2;
    float alpha = 1.0, beta = 0.0;

    // dynamically allocate
    void *A = malloc(m * k * sizeof(float));
    void *B = malloc(k * n * sizeof(float));
    float *C = malloc(m * n * sizeof(float));

    // initialize A
    float tmpA[6] = {1, 2, 3,
                      4, 5, 6};
    for (int i = 0; i < 6; i++) ((float*)A)[i] = tmpA[i];

    // initialize B
    float tmpB[6] = {7, 8,
                      9, 10,
                      11, 12};
    for (int i = 0; i < 6; i++) ((float*)B)[i] = tmpB[i];

    // initialize C
    for (int i = 0; i < m*n; i++) C[i] = 0.0;

    // call BLAS
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                alpha, A, k,
                B, n,
                beta, C, n);

    // print result
    printf("[ %f %f ]\n", C[0], C[1]);
    printf("[ %f %f ]\n", C[2], C[3]);

    free(A);
    free(B);
    free(C);

}
