#include <cblas.h>
#include "blas.h"

void dot_product(const float *A, const float *B, float *C,
                               int m, int n, int k,
                               double alpha, double beta)
{
    // Leading dimensions for ROW-MAJOR are the number of columns
    const int lda = k;
    const int ldb = n;
    const int ldc = n;

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,        // A as-is
                CblasNoTrans,        // B as-is
                m, n, k,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);
}
