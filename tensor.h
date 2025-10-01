#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

typedef struct Tensor {
    void * data;
    size_t size;
    size_t ndim;
    size_t shape[4];
    size_t stride[4];
    size_t elem_size;

    bool requires_grad;
} Tensor;

Tensor  tensor_init(const void *data, const size_t *shape, const size_t elem_size, bool requires_grad);
void    tensor_free(const Tensor *tensor);
void    tensor_print(const Tensor *tensor);

#endif
