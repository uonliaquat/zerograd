// dotprod.c
#include <stdio.h>
#include <cblas.h>   // OpenBLAS BLAS interface

int main() {
    int n = 3;
    double x[3] = {1.0, 2.0, 3.0};
    double y[3] = {4.0, 5.0, 6.0};

    // Compute dot product: x • y
    double result = cblas_ddot(n, x, 1, y, 1);

    printf("Dot product = %f\n", result);
    return 0;
}
