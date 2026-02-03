#include <cblas.h>
#include <assert.h>
#include "../include/blas.h"
#include "../include/tensor.h"

void dot_product(const double *A, const double *B, double *C,
                               int m, int n, int k)
{
    // Leading dimensions for ROW-MAJOR are the number of columns
    const int lda = k;
    const int ldb = n;
    const int ldc = n;

    double alpha = 1;
    double beta = 1;
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

Tensor dot_product_tensor(const Tensor *tensor_a, const Tensor *tensor_b)
{
    assert(tensor_a->shape[1] == tensor_b->shape[0]);

    Tensor tensor_c = tensor_init(NULL, (uint32_t[]){tensor_a->shape[0], tensor_b->shape[1]} 2, tensor_a->dtype, NULL)

    int m = (int)tensor_a->shape[0]; // No of Rows of A and C
    int n = (int)tensor_b->shape[1]; // No of Cols of B and C
    int k = (int)tensor_a->shape[1]; // No of cols of A and rows of B

    // Leading dimensions for ROW-MAJOR are the number of columns
    const int lda = k;
    const int ldb = n;
    const int ldc = n;

    double alpha = 1;
    double beta = 1;

    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,        // A as-is
                CblasNoTrans,        // B as-is
                m, n, k,
                alpha,
                ((double*)tensor_a->data), lda,
                ((double*)tensor_b->data), ldb,
                beta,
                ((double*)tensor_c.data), ldc);


    return tensor_c;
}











