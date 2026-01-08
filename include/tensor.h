#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

typedef enum DataType {
    DTYPE_INT = sizeof(int),
    DTYPE_DOUBLE = sizeof(double)
} DataType;

typedef struct Tensor {
    void * data;
    size_t size;
    size_t ndim;
    size_t shape[4];
    size_t stride[4];
    size_t elem_size;
    enum DataType dtype;

    bool requires_grad;
} Tensor;

Tensor  tensor_init(void *data, const size_t *shape, const size_t ndim, DataType dtype, const bool requires_grad, const bool random_init);
void    tensor_free(const Tensor *tensor);
Tensor  tensor_add(Tensor *tensor1, Tensor *tensor2);
void    tensor_copy_row_data(Tensor *dest_tensor, size_t dest_row, Tensor *src_tensor, size_t src_row, size_t no_of_items);
void    tensor_print(const Tensor *tensor);
void    tensor_write(const Tensor *tensor, FILE *fptr);
size_t  tensor_dtype_size(const DataType dtype);

#endif
