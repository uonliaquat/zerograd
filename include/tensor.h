#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdbool.h>
#include <stdio.h>

typedef enum DataType {
    DTYPE_INT = sizeof(int),
    DTYPE_DOUBLE = sizeof(double)
} DataType;

typedef struct Tensor {
    void * data;
    size_t size;
    size_t ndim;
    size_t shape[3];
    size_t stride[3];
    size_t elem_size;
    enum DataType dtype;
    
    bool requires_grad;
} Tensor;

Tensor  tensor_init(void *data, const size_t *shape, const size_t ndim, const DataType dtype, const bool requires_grad, const bool random_init);
void    tensor_free(const Tensor *tensor);
double  tensor_get_elem(const Tensor *tensor, size_t *coords);
void    tensor_put_elem(Tensor *tensor, size_t *coords, double elem);  
Tensor  tensor_transpose(const Tensor *tensor);
Tensor  tensor_softmax(Tensor *tensor, size_t dim);
Tensor  tensor_scale(Tensor *tensor, double elem);
Tensor  tensor_concat(Tensor *tensors, size_t no_of_tensors, size_t dim);
Tensor  *tensor_chunk(Tensor *tensor, size_t chunks, size_t dim);
Tensor  tensor_cat(Tensor **tensors, size_t len);
void    tensor_mat_mul(const Tensor *tensor1, const Tensor *tensor2, Tensor *output_tensor, size_t batch_dim);
Tensor  tensor_dot_product(const Tensor *tensor1, const Tensor *tensor2);
Tensor  tensor_add(Tensor *tensor1, Tensor *tensor2);
void    tensor_copy_row_data(Tensor *dest_tensor, size_t dest_row, Tensor *src_tensor, size_t src_row, size_t no_of_items);
void    tensor_print(const Tensor *tensor, const char *heading);
void    tensor_write_fp(const Tensor *tensor, FILE *fptr);
void    tensor_write(const Tensor *tensor, char *filename);
size_t  tensor_dtype_size(const DataType dtype);

#endif
