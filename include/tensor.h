#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>

#define TENSOR_MAX_SHAPE_DIM 8
#define TENSOR_MAX_LEN_NAME 512

typedef enum DataType {
    DTYPE_FP64,
    DTYPE_FP32,
    DTYPE_INT32,
} DataType;

typedef struct Tensor {
    char name[TENSOR_MAX_LEN_NAME];
    void * data;
    uint32_t size;
    uint8_t ndim;
    uint32_t shape[TENSOR_MAX_SHAPE_DIM];
    uint32_t stride[TENSOR_MAX_SHAPE_DIM];
    uint8_t elem_size;
    enum DataType dtype;
} Tensor;

Tensor  tensor_init(void *data, const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name);
void    tensor_init_(Tensor *tensor, void *data, const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name);
void    tensor_reset(Tensor *tensor);
Tensor  tensor_rand_init(const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name);
Tensor  tensor_file_init(FILE *fptr, uint32_t *offset, const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name);
void    tensor_free(const Tensor *tensor);
Tensor  tensor_copy(Tensor *input);
Tensor  tensor_repeat(Tensor *input, uint8_t * repeate_dims);
void    tensor_repeat_(Tensor *input, uint8_t * repeate_dims, Tensor *output);
void    tensor_unsqueeze_(Tensor *input, size_t dim);
float   tensor_get_elem(const Tensor *tensor, uint32_t *coords);
void    tensor_put_elem(Tensor *tensor, uint32_t *coords, double elem);  
Tensor  tensor_transpose(const Tensor *input);
void    tensor_transpose_(const Tensor *input, Tensor *output);
Tensor  tensor_softmax(Tensor *input, size_t dim);
void    tensor_softmax_(Tensor *input, size_t dim, Tensor *output);
Tensor  tensor_scale(Tensor *inpput, Tensor *scale);
Tensor  tensor_elementwise_scale(Tensor *inpput, double elem);
void    tensor_elementwise_scale_(Tensor *inpput, double elem, Tensor *output);
Tensor  tensor_vector_scale(Tensor *inpput, Tensor *vector);
Tensor  tensor_concat(Tensor *input, size_t no_of_tensors, size_t dim);
void    tensor_concat_(Tensor *input, size_t no_of_tensors, size_t dim, Tensor *output);
Tensor  *tensor_chunk(Tensor *input, size_t chunks, size_t dim);
void    tensor_chunk_(Tensor *input, size_t chunks, size_t dim, Tensor *output);
Tensor  tensor_cat(Tensor **tensors, size_t len);
Tensor  tensor_arange(const int start, const int end, const int steps);
void    tensor_arange_(const int start, const int end, const int steps, Tensor *output);
void    tensor_mat_mul(const Tensor *tensor1, const Tensor *tensor2, Tensor *output_tensor, size_t batch_dim);
Tensor  tensor_dot_product(const Tensor *tensor1, const Tensor *tensor2);
void    tensor_dot_product_(const Tensor *input1, const Tensor *input2, Tensor *output);
Tensor  tensor_add(Tensor *input1, Tensor *input2);
void    tensor_add_(Tensor *input1, Tensor *input2, Tensor *output);
Tensor  tensor_elementwise_add(Tensor *inpput, double val);
Tensor  tensor_vector_add(Tensor *inpput, Tensor *vector);
Tensor  tensor_tril(Tensor *input, double elem);
void    tensor_tril_(Tensor *input, double elem, Tensor *output);
void    tensor_masked_fill(Tensor *tensor, double mask, double fill);
void    tensor_copy_row_data(Tensor *dest_tensor, size_t batch_id, size_t row_id, Tensor *src_tensor, size_t src_row, size_t no_of_items);
Tensor  tensor_mean_var(Tensor *x);
Tensor  tensor_norm(Tensor *x, Tensor *mean_var_tensor, double eps);
void    tensor_print(const Tensor *tensor, const char *heading);
//void    tensor_write_fp(const Tensor *tensor, FILE *fptr);
//void    tensor_write(const Tensor *tensor, char *filename);
static inline uint8_t tensor_dtype_size(const DataType dtype){
    switch (dtype){
        case DTYPE_FP64:    return 64;
        case DTYPE_FP32:    return 32;
        case DTYPE_INT32:   return 32;
    }
}

static inline char* tensor_dtype_name(const DataType dtype){
    switch (dtype){
        case DTYPE_FP64:    return "F64";
        case DTYPE_FP32:    return "F32";
        case DTYPE_INT32:   return "I32";
        default:            return "UNKNOWN";
    }
}
#endif
