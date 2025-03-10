#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <time.h>

#include "utils.h"

#define GET_ELEMENT(tensor, row, col) \
    (tensor->dtype == FLOAT32) ? \
        (((float*)tensor->data)[row * tensor->cols + col]) : \
        (((double*)tensor->data)[row* tensor->cols + col])

#define PUT_ELEMENT(tensor, row, col, val) \
    (tensor->dtype == FLOAT32) ? \
        (((float*)tensor->data)[row * tensor->cols + col] = val) : \
        (((double*)tensor->data)[row * tensor->cols + col] = val)

typedef enum DataType{
    FLOAT32,
    FLOAT64
} DataType;

typedef struct Tensor{
    void *data;
    DataType dtype;
    size_t rows;
    size_t cols;
} Tensor;


// Tensor Functions
Tensor create_tensor(const size_t, const size_t, DataType);
void init_tensor_rand(Tensor*);
void gemm(Tensor *, Tensor *, Tensor *);
void print_tensor(Tensor *);
void save_tensor(Tensor *, char *);

#endif
