#ifndef BLAS_H
#define BLAS_H

#include "tensor.h"

void dot_product(const double *A, const double *B, double *C, int m, int n, int k);
Tensor dot_product_tensor(const Tensor *A, const Tensor *B);

#endif
