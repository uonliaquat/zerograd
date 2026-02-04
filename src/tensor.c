#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "../include/tensor.h"
#include "../include/utils.h"



static inline void shape_init(Tensor *tensor, const uint32_t *shape){
    // Initializing shape
    memset(tensor->shape, 0, sizeof(tensor->shape));
    memcpy(tensor->shape, shape, sizeof(tensor->shape));
}

static inline void size_init(Tensor *tensor){
    // Initialzing size, (total no of elements in tensor)
    tensor->size = 1;
    for(size_t i = 0; i <     tensor->ndim; i++) tensor->size = tensor->size * tensor->shape[i];
    //tensor->size = tensor->size * tensor->elem_size;
}

static inline void stride_init(Tensor *tensor){
    // Initializing stride
    for(size_t i = 0; i < tensor->ndim; i++) tensor->stride[i] = 1;
    for(int i =     tensor->ndim-2; i >= 0; i--){
        tensor->stride[i] = tensor->stride[i+1] * tensor->shape[i+1];
    }
}

static inline void name_init(Tensor *tensor, char *name){
    if(name != NULL){
        memset(tensor->name, 0, TENSOR_MAX_LEN_NAME);
        memcpy(tensor->name, name, strlen(name));
    }
}

static inline void data_init(Tensor *tensor, void *data){
    // Initializing data
    tensor->data = (void*)calloc(tensor->size,  tensor_dtype_size(tensor->dtype)/8);
    if(data && tensor->data) memcpy(tensor->data, data, tensor->size *  tensor_dtype_size(tensor->dtype)/8);
    else memset(tensor->data, 0, tensor->size);
    // printf("Sanity Check\n");
    // for(size_t i = 0; i < 10; i++){
    //     printf("%f\n", ((float*)tensor->data)[i]);
    // }
}

static inline void data_rand_init(Tensor *tensor){
    tensor->data = (void*)calloc(tensor->size,   tensor_dtype_size(tensor->dtype)/8);
    for(size_t i = 0; i < tensor->size; i++){
        double rand_number = rand_uniform(-1.0, 1.0);
        ((double*)tensor->data)[i] = rand_number;
    }
}

static inline void data_file_init(Tensor *tensor, FILE *fptr, const uint32_t *offset){
    tensor->data = (void*)calloc(tensor->size,   tensor_dtype_size(tensor->dtype)/8);
    for(size_t i = 0; i < tensor->size; i++){
        float elem = 0;
        ((float*)tensor->data)[i] = 0;
    }
}

Tensor tensor_init(void *data, const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name){

    Tensor tensor;
    tensor_reset(&tensor);

    tensor.dtype = dtype;
    tensor.ndim = ndim;
    tensor.elem_size = tensor_dtype_size(dtype);
    

    shape_init(&tensor, shape);
    size_init(&tensor);
    stride_init(&tensor);
    name_init(&tensor, name);
    data_init(&tensor, data);
    return tensor;
}

void tensor_init_(Tensor *tensor, void *data, const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name){

    tensor_reset(tensor);

    tensor->dtype = dtype;
    tensor->ndim = ndim;
    tensor->elem_size = tensor_dtype_size(dtype);
    

    shape_init(tensor, shape);
    size_init(tensor);
    stride_init(tensor);
    name_init(tensor, name);
    data_init(tensor, data);
}

void tensor_reset(Tensor *tensor){
    if(tensor != NULL){
        tensor->dtype = 0;
        tensor->ndim = 0;
        tensor->elem_size = 0;
        tensor->size = 0;
        tensor->data = NULL;
        memset(tensor->name, 0, sizeof(tensor->name));
        memset(tensor->shape, 0, sizeof(tensor->shape));
        memset(tensor->stride, 0, sizeof(tensor->stride));
    }
}

Tensor tensor_rand_init(const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name){

    Tensor tensor = { 0 };

    tensor.dtype = dtype;
    tensor.ndim = ndim;
    tensor.elem_size = tensor_dtype_size(dtype);

    shape_init(&tensor, shape);
    size_init(&tensor);
    stride_init(&tensor);
    name_init(&tensor, name);
    data_rand_init(&tensor);
    return tensor;
    
}
// Tensor tensor_file_init(FILE *fptr, uint32_t *offset, const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name){

//     Tensor tensor = { 0 };

//     tensor.dtype = dtype;
//     tensor.ndim = ndim;
//     tensor.elem_size = dtype_size(dtype);

//     shape_init(&tensor, shape, ndim);
//     stride_init(&tensor, ndim);
//     size_init(&tensor, ndim);
//     name_init(&tensor, name);
//     data_file_init(&tensor, fptr, offset);
//     return tensor;
    
// }

void tensor_free(const Tensor *tensor){
    if(tensor->size > 0) free(tensor->data);
}

static inline size_t tensor_get_batch_size(const Tensor *input){
    return input->ndim == 3 ? input->shape[0]: 1;
}

static inline size_t tensor_get_rows(const Tensor *input){
    return input->ndim == 3 ? input->shape[1]: input->shape[0];
}

static inline size_t tensor_get_cols(const Tensor *input){
    return input->ndim == 3 ? input->shape[2]: input->shape[1];
}

Tensor tensor_copy(Tensor *input)
{
    Tensor output_tensor = tensor_init(
        NULL, 
        input->shape,
        input->ndim,
        input->dtype,
        NULL
    );
    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                double elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, elem);
            }
        }
    }
    return output_tensor;
}


Tensor tensor_repeat(Tensor *input, uint8_t * repeate_dims){
     Tensor output_tensor = tensor_init(
        NULL, 
        (uint32_t[]){repeate_dims[0] == 1 ? input->shape[0] : repeate_dims[0], repeate_dims[1] == 1 ? input->shape[1] : repeate_dims[1] },
        input->ndim,
        input->dtype,
        NULL
    );
    // Initializing stride
    input->stride[0] = 1;
    input->stride[1] = 1;
    input->stride[2] = 1;
    //memset(tensor.stride, 0, sizeof(tensor.stride));
    //ndim = 3
    for(int i = input->ndim-2; i >= 0; i--){
        input->stride[i] = input->stride[i+1] * input->shape[i+1];
    }

    if(repeate_dims[0] != 0){
        for(size_t i = 0; i < output_tensor.shape[0]; i++){
            //printf("%zu, %zu\n", i, input->size);
            tensor_copy_row_data(&output_tensor, i, 0, input, 0, output_tensor.size*tensor_dtype_size(output_tensor.dtype));
        }
    }
    return output_tensor;
}

void tensor_repeat_(Tensor *input, uint8_t * repeate_dims, Tensor *output){
    if(output->data == NULL){
        *output = tensor_init(
            NULL, 
            (uint32_t[]){repeate_dims[0] == 1 ? input->shape[0] : repeate_dims[0], repeate_dims[1] == 1 ? input->shape[1] : repeate_dims[1], repeate_dims[2] == 1 ? input->shape[2] : repeate_dims[2]  },
            input->ndim,
            input->dtype,
            NULL
        );
    }
    // Initializing stride
    input->stride[0] = 1;
    input->stride[1] = 1;
    input->stride[2] = 1;
    //memset(tensor.stride, 0, sizeof(tensor.stride));
    //ndim = 3
    for(int i = input->ndim-2; i >= 0; i--){
        input->stride[i] = input->stride[i+1] * input->shape[i+1];
    }
    if(repeate_dims[0] != 0){
        for(size_t i = 0; i < output->shape[0]; i++){
            //printf("%zu, %zu\n", i,   input->size*tensor_dtype_size(output->dtype));
            tensor_copy_row_data(output, i, 0, input, 0, input->size*tensor_dtype_size(output->dtype));
        }
    }
 }

void tensor_unsqueeze_(Tensor *input, size_t dim){
    assert(dim == 0);

    input->ndim += 1;

    for(size_t i = input->ndim; i > dim; i--){
        input->shape[i] = input->shape[i-1];
    }
    input->shape[dim] = 1;

    input->stride[0] = 1;
    input->stride[1] = 1;
    input->stride[2] = 1;
    // memset(tensor.stride, 0, sizeof(tensor.stride));
    // //ndim = 3
    for(int i = input->ndim-2; i >= 0; i--){
        input->stride[i] = input->stride[i+1] * input->shape[i+1];
    }
}

float tensor_get_elem(const Tensor *tensor, uint32_t *coords){
    assert(tensor->dtype == DTYPE_FP32 || tensor->dtype == DTYPE_INT32);
    size_t index = 0;
    for(size_t d = 0; d < tensor->ndim; d++){
        index += coords[d] * tensor->stride[d];
    }
    if(tensor->dtype == DTYPE_FP32)
        return ((float*)tensor->data)[index];
    else if(tensor->dtype == DTYPE_INT32){
        return ((uint32_t*)tensor->data)[index];
    }
    return 0;
}

void tensor_put_elem(Tensor *tensor, uint32_t *coords, double elem){
    size_t index = 0;
    for(size_t d = 0; d < tensor->ndim; d++){
        index += coords[d] * tensor->stride[d];
    }
    //printf("\nindex: %zu, tensor->size: %zu\n", index, tensor->size);
    assert(index < tensor->size);
    if(tensor->dtype == DTYPE_FP64){
        ((double*)tensor->data)[index] = elem;
    }
    else if(tensor->dtype == DTYPE_INT32){
        ((int*)tensor->data)[index] = (int)elem;
        //printf("%d\n", ((int*)tensor->data)[index]);
    }
}

// Tensor tensor_transpose(const Tensor *input){
//     Tensor output = tensor_init(
//         NULL, 
//         input->ndim == 3 ? (uint32_t[]){input->shape[0], input->shape[2], input->shape[1]}: (uint32_t[]){input->shape[1], input->shape[0]}, 
//         input->ndim,
//         input->dtype,
//         NULL
//     );


//     size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
//     size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
//     size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];

//     for(size_t b = 0; b < batch_size; b++){
//         for(size_t i = 0; i < rows; i++){
//             for(size_t j = 0; j < cols; j++){
//                 double elem = tensor_get_elem(input, input->ndim ? (uint32_t[]){b, i, j} : (uint32_t[]){i, j});
//                 tensor_put_elem(&output, input->ndim ? (uint32_t[]){b, j, i}: (uint32_t[]){j, i}, elem);
//             }
//         }
//     }
//     return output;
// }

void tensor_transpose_(const Tensor *input, Tensor *output){
    if(output->size == 0){
        *output = tensor_init(
            NULL, 
            input->ndim == 3 ? (uint32_t[]){input->shape[0], input->shape[2], input->shape[1]}: (uint32_t[]){input->shape[1], input->shape[0]}, 
            input->ndim,
            input->dtype,
            NULL
        );
    }

    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];

    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                double elem = tensor_get_elem(input, input->ndim ? (uint32_t[]){b, i, j} : (uint32_t[]){i, j});
                tensor_put_elem(output, input->ndim ? (uint32_t[]){b, j, i}: (uint32_t[]){j, i}, elem);
            }
        }
    }

}


// Tensor tensor_softmax(Tensor *input, size_t dim){
//     Tensor output = tensor_init(
//         NULL, 
//         input->shape, 
//         input->ndim,
//         input->dtype,
//         true
//     );
    
//     size_t batch_size   = tensor_get_batch_size(input);
//     size_t rows         = tensor_get_rows(input);
//     size_t cols         = tensor_get_cols(input);

//     if(dim == 1){
//         for(size_t b = 0; b < batch_size; b++){
//             for(size_t i = 0; i < rows; i++){
//                 double exp_sum = 0;
//                 for(size_t j = 0; j < cols; j++){
//                     double elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
//                     exp_sum = exp_sum + expf(elem);
//                 }
//                 for(size_t j = 0; j < cols; j++){
//                     double elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});

//                     double new_elem = expf(elem) / exp_sum;
//                     tensor_put_elem(&output, output.ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
//                 }
//             }
//         }
//     }
//     return output;
// }
void tensor_softmax_(Tensor *input, size_t dim, Tensor *output){
    if(output->size == 0){
        *output = tensor_init(
            NULL, 
            input->shape, 
            input->ndim,
            input->dtype,
            NULL
        );
    }

    size_t batch_size   = tensor_get_batch_size(input);
    size_t rows         = tensor_get_rows(input);
    size_t cols         = tensor_get_cols(input);

    if(dim == 1){
        for(size_t b = 0; b < batch_size; b++){
            for(size_t i = 0; i < rows; i++){
                double exp_sum = 0;
                
                for(size_t j = 0; j < cols; j++){
                    double elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                    exp_sum = exp_sum + expf(elem);
                }
                for(size_t j = 0; j < cols; j++){
                    double elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});

                    double new_elem = expf(elem) / exp_sum;
                    tensor_put_elem(output, output->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
                }
            }
        }
    }
}


// // Tensor tensor_scale(Tensor *input, Tensor *scale){
// //     assert(scale->ndim == 2);
// //     Tensor output_tensor = tensor_init(
// //         NULL, 
// //         input->shape, 
// //         input->ndim, 
// //         input->dtype, 
// //         false
// //     );

// //     size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
// //     size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
// //     size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


// //     for(size_t i = 0; i < batch_size; i++){
// //         for(size_t j = 0; j < rows; j++){
// //             double scale_factor =  tensor_get_elem(scale, (uint32_t[]){j, i});
// //             for(size_t k = 0; k < cols; k++){
// //                 double old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k});
// //                 double new_elem = old_elem * scale_factor;
// //                 tensor_put_elem(&output_tensor, input->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k}, new_elem);
// //             }
// //         }
// //     }
// //     return output_tensor;
// // }

// Tensor tensor_elementwise_scale(Tensor *input, double elem){
//     Tensor output = tensor_init(
//         NULL, 
//         input->shape,
//         input->ndim,
//         input->dtype,
//         NULL
//     );


//     size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
//     size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
//     size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
//     for(size_t b = 0; b < batch_size; b++){
//         for(size_t i = 0; i < rows; i++){
//             for(size_t j = 0; j < cols; j++){
//                 double old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
//                 double new_elem = old_elem * elem;
//                 tensor_put_elem(&output,  output.ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
//             }
//         }
//     }
//     return output;
// }

void tensor_elementwise_scale_(Tensor *input, double elem, Tensor *output){
    if(output->size == 0){
        *output = tensor_init(
            NULL, 
            input->shape,
            input->ndim,
            input->dtype,
            NULL
        );
    }
    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                double old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                double new_elem = old_elem * elem;
                tensor_put_elem(output,  output->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
            }
        }
    }

}
Tensor tensor_vector_scale(Tensor *input, Tensor *vector){
    assert(vector->ndim == 2);
    Tensor output_tensor = tensor_init(
        NULL, 
        input->shape, 
        input->ndim, 
        input->dtype, 
        false
    );

    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            double scale_factor =  tensor_get_elem(vector, (uint32_t[]){i, b});
            for(size_t j = 0; j < cols; j++){
                double old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                double new_elem = old_elem * scale_factor;
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
            }
        }
    }
    return output_tensor;
}

// Tensor tensor_concat(Tensor *input, size_t no_of_tensors, size_t dim){
//     //this function doesn't work on batch of tensors
//     if(dim == 1){
//         size_t new_cols = input[0].shape[input[0].ndim -1];
//         for(size_t i = 1; i < no_of_tensors; i++){
//             printf("no_of_tensors: %zu\n", no_of_tensors);
//             printf("input[i-1]->ndim %zu, input[i]->ndim: %zu\n", input[i-1].ndim, input[i].ndim);
//             assert(input[i-1].ndim == input[i].ndim);
//             assert(input[i-1].shape[0] == input[i].shape[0]);
//             assert(input[i-1].shape[1] == input[i].shape[1]);
//             if(input[i-1].ndim == 3){
//                 assert(input[i-1].shape[2] == input[2].shape[2]);
//             }
//             new_cols += input[i].shape[input[i].ndim -1];
//         }
//         //concat across columns
//         size_t batch_size   = tensor_get_batch_size(input);
//         size_t rows         = tensor_get_rows(input);
//         size_t cols         = tensor_get_cols(input);
//         // // printf("===================================================================================\n");
//         // // printf("batch_size: %zu, rows: %zu, cols: %zu, new_cols: %zu\n", batch_size, rows, cols, new_cols);
//         // // printf("===================================================================================\n");

//         // size_t new_cols = cols;
//         // for(size_t i = 1; i < no_of_tensors; i++){
//         //     // if(input[i]->ndim == 2) assert(input[i]->shape[0] == input[i-1]->shape[0]);
//         //     // else if(input[i]->ndim == 3) assert(input[i]->shape[1] == input[i-1]->shape[1]);
//         //     new_cols += input[i]->shape[1];
//         // }
 
//         Tensor output = tensor_init(
//             NULL, 
//             (uint32_t[]){batch_size, rows, new_cols}, 
//             input[0].ndim,
//             input[0].dtype,
//             input[0].requires_grad,
//             false
//         );
//         for(size_t b = 0; b < batch_size; b++){
//             for(size_t t = 0; t < no_of_tensors; t++){
//                 for(size_t i = 0; i < rows; i++){
//                     for(size_t j = 0; j < cols; j++){
//                         double elem = tensor_get_elem(&input[t], input[t].ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
//                         tensor_put_elem(&output,  output.ndim == 3 ? (uint32_t[]){b, i, j + (t * cols)}: (uint32_t[]){i, j + (t * cols)}, elem);
//                     }
//                 }
//             }
//         }
//         return output;
//     }
//     return (Tensor){};
// }

void tensor_concat_(Tensor *input, size_t no_of_tensors, size_t dim, Tensor *output){
    //this function doesn't work on batch of tensors
    if(dim == 1){
        size_t new_cols = input[0].shape[input[0].ndim -1];
        for(size_t i = 1; i < no_of_tensors; i++){
            //printf("no_of_tensors: %zu\n", no_of_tensors);
            //printf("input[i-1]->ndim %zu, input[i]->ndim: %zu\n", input[i-1].ndim, input[i].ndim);
            assert(input[i-1].ndim == input[i].ndim);
            assert(input[i-1].shape[0] == input[i].shape[0]);
            assert(input[i-1].shape[1] == input[i].shape[1]);
            // if(input[i-1].ndim == 3){
            //     assert(input[i-1].shape[2] == input[2].shape[2]);
            // }
            new_cols += input[i].shape[input[i].ndim -1];
        }
        //concat across columns
        size_t batch_size   = tensor_get_batch_size(input);
        size_t rows         = tensor_get_rows(input);
        size_t cols         = tensor_get_cols(input);
        // // printf("===================================================================================\n");
        // // printf("batch_size: %zu, rows: %zu, cols: %zu, new_cols: %zu\n", batch_size, rows, cols, new_cols);
        // // printf("===================================================================================\n");

        // size_t new_cols = cols;
        // for(size_t i = 1; i < no_of_tensors; i++){
        //     // if(input[i]->ndim == 2) assert(input[i]->shape[0] == input[i-1]->shape[0]);
        //     // else if(input[i]->ndim == 3) assert(input[i]->shape[1] == input[i-1]->shape[1]);
        //     new_cols += input[i]->shape[1];
        // }
        if(output->size == 0){
            *output = tensor_init(
                NULL, 
                (uint32_t[]){batch_size, rows, new_cols}, 
                input[0].ndim,
                input[0].dtype,
                NULL
            );
        }
        for(size_t b = 0; b < batch_size; b++){
            for(size_t t = 0; t < no_of_tensors; t++){
                for(size_t i = 0; i < rows; i++){
                    for(size_t j = 0; j < cols; j++){
                        double elem = tensor_get_elem(&input[t], input[t].ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                        tensor_put_elem(output,  output->ndim == 3 ? (uint32_t[]){b, i, j + (t * cols)}: (uint32_t[]){i, j + (t * cols)}, elem);
                    }
                }
            }
        }
    }
}

Tensor *tensor_chunk(Tensor *input, size_t chunks, size_t dim){
    assert(input->ndim <= 3);
    Tensor *output_tensors = calloc(chunks, sizeof(Tensor));
    if(dim == 1){
        assert(input->shape[input->ndim - 1] % chunks == 0);
        //split columns
        size_t cols = input->shape[input->ndim - 1] / chunks;

        printf("chunks: %zu\n", chunks);
        size_t batch_size = tensor_get_batch_size(input);
        size_t rows = tensor_get_rows(input);
        //size_t cols = tensor_get_cols(input);
        for(size_t chunk = 0; chunk < chunks; chunk++){
            Tensor output_tensor = tensor_init(
                NULL, 
                input->ndim == 3 ? (uint32_t[]){input->shape[0], input->shape[1], cols} : (uint32_t[]){input->shape[0], cols}, 
                input->ndim,
                input->dtype,
                NULL
            );
            
            printf("batch_size: %zu, rows: %zu, cols: %zu\n", batch_size, rows, cols);
            for(size_t b = 0; b < batch_size; b++){
                for(size_t i = 0; i < rows; i++){
                    for(size_t j = 0; j < cols; j++){
                        double elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j + (chunk * cols)}: (uint32_t[]){i, j + (chunk * cols)});
                        tensor_put_elem(&output_tensor, output_tensor.ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){b, i}, elem);
                    }
                }
            }
            output_tensors[chunk] = output_tensor;
        }
    }
    return output_tensors;
}

void tensor_chunk_(Tensor *input, size_t chunks, size_t dim, Tensor *output){
    printf("\n\n\n\n\n ******************* CHUNK ****************\n");
    //tensor_print(input, "input");
    assert(input->ndim <= 3);
    if(dim == 1){
        assert(input->shape[input->ndim - 1] % chunks == 0);
        //split columns
        size_t cols = input->shape[input->ndim - 1] / chunks;

        size_t batch_size = tensor_get_batch_size(input);
        size_t rows = tensor_get_rows(input);
        //printf("input->shape[input->ndim - 1]: %zu, chunks: %zu\n", input->shape[input->ndim - 1], chunks);
        //printf("batch_size: %zu, rows: %zu, cols: %zu\n", batch_size, rows, cols);
        //size_t cols = tensor_get_cols(input);
        //printf("chunks: %zu\n", chunks);

        for(size_t chunk = 0; chunk < chunks; chunk++){
            
            if(output[chunk].size == 0){
                output[chunk] = tensor_init(
                    NULL, 
                    input->ndim == 3 ? (uint32_t[]){input->shape[0], input->shape[1], cols} : (uint32_t[]){input->shape[0], cols}, 
                    input->ndim,
                    input->dtype,
                    NULL
                );
            }
    
            for(size_t b = 0; b < batch_size; b++){
                for(size_t i = 0; i < rows; i++){
                    for(size_t j = 0; j < cols; j++){
                        double elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j + (chunk * cols)}: (uint32_t[]){i, j + (chunk * cols)});
                        tensor_put_elem(&output[chunk], output[chunk].ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){b, i}, elem);
                    }
                }
            }
            tensor_print(&output[chunk], " output[chunk]");
        }
    }
}


Tensor tensor_cat(Tensor **tensors, size_t len){
    
    Tensor output_tensor = tensor_init(
        NULL, 
        (uint32_t[]){tensors[0]->shape[0], tensors[0]->shape[1]*len}, 
        2,
        tensors[0]->dtype,
        NULL
    );
    size_t out_col = 0;
    for(size_t i = 0; i < len; i++){
        for(size_t j = 0; j < tensors[i]->shape[0]; j++){
            for(size_t k = 0; k < tensors[i]->shape[1]; k++){
                double elem = tensor_get_elem(tensors[i], (uint32_t[]){j, k});
                tensor_put_elem(&output_tensor, (uint32_t[]){j, k+out_col}, elem);
            }
        }
        out_col += tensors[i]->shape[1];
    }
    return output_tensor;
}

// Tensor tensor_arange(const int start, const int end, const int steps){
//     //returns 1d tensor only
//     Tensor output_tensor = tensor_init(
//         NULL, 
//         (uint32_t[]){1, (int)((end - start)/steps)}, 
//         2,
//         DTYPE_INT32,
//         NULL
//     );
//     for(size_t i = start; i <=end; i += steps){
//         tensor_put_elem(&output_tensor, (uint32_t[]){0, (double)i}, i);
//     }
//     return output_tensor;
// }
void tensor_arange_(const int start, const int end, const int steps, Tensor *output){
    if(output->data == NULL){
        *output = tensor_init(
            NULL, 
            (uint32_t[]){1, (int)((end - start)/steps)}, 
            2,
            DTYPE_INT32,
            NULL
        );
    }
    for(size_t i = start; i < end; i += steps){
        //printf("%zu\n", i);
        tensor_put_elem(output, (uint32_t[]){0, i}, i);
    }
}


void tensor_mat_mul(const Tensor *tensor1, const Tensor *tensor2, Tensor *output_tensor, size_t batch_dim){


    size_t t1_batch_size = tensor1->ndim == 3 ? tensor1->shape[0]: 1;
    size_t t1_rows = tensor1->ndim == 3 ? tensor1->shape[1]: tensor1->shape[0];
    size_t t1_cols = tensor1->ndim == 3 ? tensor1->shape[2]: tensor1->shape[1];

    size_t t2_batch_size = tensor2->ndim == 3 ? tensor2->shape[0]: 1;
    size_t t2_rows = tensor2->ndim == 3 ? tensor2->shape[1]: tensor2->shape[0];
    size_t t2_cols = tensor2->ndim == 3 ? tensor2->shape[2]: tensor2->shape[1];

    //size_t t2_batch_size = tensor2->ndim == 3 ? tensor2->shape[0]: 1;
    size_t out_rows = output_tensor->ndim == 3 ? output_tensor->shape[1]: output_tensor->shape[0];
    size_t out_cols = output_tensor->ndim == 3 ? output_tensor->shape[2]: output_tensor->shape[1];

    for(size_t i = 0; i < out_rows; i++){
        for(size_t j = 0; j < out_cols; j++){
            double result = 0;
            for(size_t k = 0; k < t1_cols; k++){
                double elem1 = tensor_get_elem(tensor1, tensor1->ndim == 3 ?(uint32_t[]){batch_dim, i, k}:  (uint32_t[]){i, k});
                double elem2 = tensor_get_elem(tensor2, tensor2->ndim == 3 ?(uint32_t[]){batch_dim, k, j}: (uint32_t[]){k, j});
                result += elem1 * elem2;
            }
            if(output_tensor->ndim == 3) tensor_put_elem(output_tensor, (uint32_t[]){batch_dim, i, j}, result);
            else if(output_tensor->ndim == 2)  tensor_put_elem(output_tensor, tensor2->ndim == 3 ? (uint32_t[]){batch_dim, i, j}: (uint32_t[]){i, j}, result);
        }
    }
}


// Tensor tensor_dot_product(const Tensor *input1, const Tensor *input2){
//     //assert(tensor1->ndim == tensor2->ndim);
//     // tensor_print(tensor1, "================== TENSOR 1 ==================");
//     // tensor_print(tensor2, "================== TENSOR 2 ==================");
//     size_t t1_batch_size = tensor_get_batch_size(input1);
//     size_t t1_rows = tensor_get_rows(input1);
//     size_t t1_cols = tensor_get_cols(input1);

//     size_t t2_batch_size = tensor_get_batch_size(input2);
//     size_t t2_rows = tensor_get_rows(input2);
//     size_t t2_cols = tensor_get_rows(input2);

//     //printf("t1_cols: %zu, t2_rows: %zu\n", t1_cols, t2_rows);
//     assert(t2_batch_size == 1);
//     assert(t1_cols == t2_rows);

//     Tensor output =  tensor_init(
//         NULL, 
//         (uint32_t[]){t1_batch_size, t1_rows, t2_cols}, input1->ndim,
//         input1->dtype,
//         input1->requires_grad,
//         NULL
//     );
//     for(size_t b = 0; b < t1_batch_size; b++){
//         tensor_mat_mul(input1, input2, &output, b);
//     }
//     return output;
// }

void tensor_dot_product_(const Tensor *input1, const Tensor *input2, Tensor *output){
    //assert(tensor1->ndim == tensor2->ndim);
    // tensor_print(tensor1, "================== TENSOR 1 ==================");
    // tensor_print(tensor2, "================== TENSOR 2 ==================");
    size_t t1_batch_size = tensor_get_batch_size(input1);
    size_t t1_rows = tensor_get_rows(input1);
    size_t t1_cols = tensor_get_cols(input1);

    size_t t2_batch_size = tensor_get_batch_size(input2);
    size_t t2_rows = tensor_get_rows(input2);
    size_t t2_cols = tensor_get_cols(input2);

    // assert(t2_batch_size == 1);
    //printf("t1_rows: %zu, t1_cols: %zu, t2_rows: %zu, t2_cols: %zu\n", t1_rows, t1_cols, t2_rows, t2_cols);
    assert(t1_cols == t2_rows);

    
    if(output->size == 0){
        *output = tensor_init(
            NULL, 
            (uint32_t[]){t1_batch_size, t1_rows, t2_cols}, 
            input1->ndim,
            input1->dtype,
            NULL
        );
    }
    for(size_t b = 0; b < t1_batch_size; b++){
        tensor_mat_mul(input1, input2, output, b);
    }

}





// // Tensor tensor_dot_product(const Tensor *tensor1, const Tensor *tensor2){
// //     assert(tensor1->shape[1] == tensor2->shape[0]);
// //     Tensor output_tensor =  tensor_init(
// //         NULL, 
// //         (uint32_t[]){tensor1->shape[0],  
// //         tensor2->shape[1]}, 
// //         2,
// //         tensor1->dtype,
// //         NULL
// //     );
// //     size_t t1_rows = tensor1->shape[0];
// //     size_t t1_cols = tensor1->shape[1];
// //     size_t t2_rows = tensor2->shape[0];
// //     size_t t2_cols = tensor2->shape[1];
// //     size_t out_rows = output_tensor.shape[0];
// //     size_t out_cols = output_tensor.shape[1];
// //     for(size_t i = 0; i < out_rows; i++){
// //         for(size_t j = 0; j < out_cols; j++){
// //             double result = 0;
// //             for(size_t k = 0; k < t1_cols; k++){
// //                 double elem1 = tensor_get_elem(tensor1, (uint32_t[]){i, k});
// //                 double elem2 = tensor_get_elem(tensor2, (uint32_t[]){k, j});
// //                 result += elem1 * elem2;
// //             }
// //             tensor_put_elem(&output_tensor, i, j, result);
// //         }
// //     }
// //     return output_tensor;
// // }

// Tensor tensor_add(Tensor *input1, Tensor *input2){
//     Tensor output = tensor_init(
//         input1->data, 
//         input1->shape, 
//         input1->ndim, 
//         input1->dtype, 
//         NULL
//     );
//     for(size_t i = 0; i < output.size; i++){
//         ((double*)output.data)[i] = ((double*)output.data)[i] + ((double*)input2->data)[i];
//     }
//     return output;
// }

void tensor_add_(Tensor *input1, Tensor *input2, Tensor *output){
    assert(input1->dtype == input2->dtype);
    if(output->size == 0){
        *output = tensor_init(
            input1->data, 
            input1->shape, 
            input1->ndim, 
            input1->dtype,
            NULL
        );
    }
    memcpy(output->data, input1->data, input1->size * tensor_dtype_size(input1->dtype));
    for(size_t i = 0; i < output->size; i++){
        ((float*)output->data)[i] = ((float*)output->data)[i] + ((float*)input2->data)[i];
    }
}

Tensor tensor_elementwise_add(Tensor *tensor, double val){

    Tensor outout_tensor = tensor_init(
        tensor->data, 
        tensor->shape, 
        tensor->ndim, 
        tensor->dtype, 
        NULL
    );

    size_t batch_size = tensor->ndim == 3 ? tensor->shape[0]: 1;
    size_t rows = tensor->ndim == 3 ? tensor->shape[1]: tensor->shape[0];
    size_t cols = tensor->ndim == 3 ? tensor->shape[2]: tensor->shape[1];

    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(tensor, tensor->ndim ==3 ? (uint32_t[]){i, j, k}:  (uint32_t[]){j, k});
                tensor_put_elem(tensor, tensor->ndim ==3 ? (uint32_t[]){i, j, k}:  (uint32_t[]){j, k}, val * elem);
            }
        }
    }
    return outout_tensor;
}


Tensor tensor_vector_add(Tensor *input, Tensor *vector){
    assert(vector->ndim == 2);
    Tensor output_tensor = tensor_init(
        NULL, 
        input->shape, 
        input->ndim, 
        input->dtype,
        NULL
    );

    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double scale_factor =  tensor_get_elem(vector, (uint32_t[]){j, i});
            for(size_t k = 0; k < cols; k++){
                double old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k});
                double new_elem = old_elem + scale_factor;
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k}, new_elem);
            }
        }
    }
    return output_tensor;
}

Tensor tensor_tril(Tensor *input, double elem){
    //assert(ndim == 2);
    Tensor output = tensor_init(
        NULL, 
        input->shape, 
        input->ndim, 
        input->dtype, 
        NULL
    );


    // Tensor new_tensor = tensor_init(NULL, shape, ndim, dtype, false, false);
    for(size_t i = 0; i < input->shape[0]; i++){
        for(size_t j = 0; j < input->shape[1]; j++){
            if(j > i) tensor_put_elem(input, (uint32_t[]){i,j}, elem);
            // else tensor_put_elem(&new_tensor, (uint32_t[]){i,j}, 0);
        }
    }
    return output;
}

// void tensor_tril_(Tensor *input, double elem, Tensor *output){
//     if(output->size == 0){

//     }
// }

void tensor_masked_fill(Tensor *tensor, double mask, double fill){
    assert(tensor->ndim == 2);
    for(size_t i = 0; i < tensor->shape[0]; i++){
        for(size_t j = 0; j < tensor->shape[1]; j++){
            double elem = tensor_get_elem(tensor, (uint32_t[]){i, j});
            if(elem == mask){
                tensor_put_elem(tensor, (uint32_t[]){i, j}, fill);
            }
        }
    }
}



void tensor_copy_row_data(Tensor *dest_tensor, size_t batch_id, size_t row_id, Tensor *src_tensor, size_t src_row, size_t no_of_items){
    size_t dest_index = batch_id * dest_tensor->stride[0] + row_id * dest_tensor->stride[1];
    size_t src_index =  src_row * src_tensor->stride[0];
    //printf("batch_id %zu, dest_tensor->stride[0]: %u, row_id: %zu,  dest_tensor->stride[1]: %u\n", batch_id, dest_tensor->stride[0], row_id, dest_tensor->stride[1]);
    // printf("dest index %zu, ", dest_index);
    // printf("src_index: %zu\n", src_index);
    void *dest, *src;
    if(dest_tensor->dtype == DTYPE_FP32){
        dest = &((float*)dest_tensor->data)[dest_index];
        src =  &((float*)src_tensor->data)[src_index];
        memcpy(dest, src, no_of_items * src_tensor->elem_size);
    }
    else if(dest_tensor->dtype == DTYPE_INT32){
        dest = &((int*)dest_tensor->data)[dest_index];
        src =  &((int*)src_tensor->data)[src_index];
        memcpy(dest, src, no_of_items * src_tensor->elem_size);
    }
}

Tensor tensor_mean_var(Tensor *x){

    Tensor output_tensor = tensor_init(
        NULL, 
        x->ndim ==3 ? (uint32_t[]){x->shape[0], x->shape[1], 2}:  (uint32_t[]){x->shape[0], 2}, 
        x->ndim, 
        x->dtype,  
        NULL
    );
    size_t batch_size = x->ndim == 3 ? x->shape[0]: 1;
    size_t rows = x->ndim == 3 ? x->shape[1]: x->shape[0];
    size_t cols = x->ndim == 3 ? x->shape[2]: x->shape[1];
    //printf("batch_size: %zu, rows: %zu, cols: %zu, x->ndim: %zu\n", batch_size, rows, cols, x->ndim);
    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double mean = 0;
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(x, x->ndim ==3 ? (uint32_t[]){i, j, k}:  (uint32_t[]){j, k});
                mean += elem;
            }
            mean = mean / cols;
            printf("%f\n", mean);

            double variance = 0;
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(x, x->ndim ==3 ? (uint32_t[]){i, j, k}:  (uint32_t[]){j, k});
                double squared_deviation = pow(elem - mean, 2);
                //double sqrt_squared_deviation = sqrt(squared_deviation);
                variance += squared_deviation;
                //printf("elem: %.3f, mean: %.3f, elem-mean: %.3f, squared_deviation: %.3f, sqrt_squared_deviation: %.3f\n", elem, mean, elem - mean, squared_deviation, sqrt_squared_deviation);
            }
            variance = variance / cols; 
            tensor_put_elem(&output_tensor, x->ndim ==3 ? (uint32_t[]){i, j, 0}:  (uint32_t[]){j, 0}, mean);
            tensor_put_elem(&output_tensor, x->ndim ==3 ? (uint32_t[]){i, j, 1}:  (uint32_t[]){j, 1}, variance);
        }
    }
    return output_tensor;
}

Tensor tensor_norm(Tensor *x, Tensor *mean_var_tensor, double eps){
    assert(x->ndim == mean_var_tensor->ndim);
    assert(mean_var_tensor->shape[mean_var_tensor->ndim-1] == 2);
    assert(x->shape[0] == mean_var_tensor->shape[0]);
    if(x->ndim == 3) assert(x->shape[1] == mean_var_tensor->shape[1]);

    
    Tensor output_tensor = tensor_init(
        NULL, 
        x->shape,
        x->ndim, 
        x->dtype, 
        NULL
    );

    size_t batch_size = x->ndim == 3 ? x->shape[0]: 1;
    size_t rows = x->ndim == 3 ? x->shape[1]: x->shape[0];
    size_t cols = x->ndim == 3 ? x->shape[2]: x->shape[1];
    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double mean = tensor_get_elem(mean_var_tensor, x->ndim == 3 ? (uint32_t[]){i, j, 0}: (uint32_t[]){j, 0});
            double var = tensor_get_elem(mean_var_tensor, x->ndim == 3 ? (uint32_t[]){i, j, 1}: (uint32_t[]){j, 1});
            //printf("mean: %.2f, var: %.2f\n", mean, var);
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(x, x->ndim == 3? (uint32_t[]){i, j, k}: (uint32_t[]){j, k});
                double norm_elem = (elem - mean) / sqrt(var + eps);
                tensor_put_elem(&output_tensor, x->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k}, norm_elem);
            }
        }
    }
    return output_tensor;
}

void tensor_print(const Tensor *tensor, const char *heading){
    printf("\n============== %s ==================\n", heading);
    printf("size:           %u\n", tensor->size);
    printf("ndim:           %u\n", tensor->ndim);
    
    printf("shape:          ( ");
    for(size_t i = 0; i < tensor->ndim; i++){
        printf("%u, ", tensor->shape[i]);
    }
    printf(" )\n");
    printf("stride:         ( ");
    for(size_t i = 0; i < tensor->ndim; i++){
        printf("%u, ", tensor->stride[i]);
    }
    printf(" )\n");
    printf("elem_size:      %u\n", tensor->elem_size);
    printf("data:\n");

    if(tensor->ndim == 4){
        // uint32_t shape_i = tensor->shape[0];
        // uint32_t shape_j = tensor->shape[1];
        // uint32_t shape_k = tensor->shape[2];
        // uint32_t shape_l = tensor->shape[3];

        uint32_t shape_i = 1;
        uint32_t shape_j = 1;
        uint32_t shape_k = 4;
        uint32_t shape_l = 4;
        printf("[\n");
        for(size_t i = 0; i < shape_i; i++){
            printf("    [\n");
            for(size_t j = 0; j < shape_j; j++){
                printf("        [\n");
                for(size_t k = 0; k < shape_k; k++){
                    //printf("            [\n");
                    for(size_t l = 0; l < shape_l; l++){
                        float elem = tensor_get_elem(tensor, (uint32_t[]){i, j, k, l});
                        if(tensor->dtype == DTYPE_FP32) printf("            %5.2f ", elem);
                        else if(tensor->dtype == DTYPE_INT32) printf("      %d ", (int)elem);
                    }
                    printf("\n");
                }
                printf("        ]\n");
            }
            printf("    ]\n");
        }
        printf(" ]\n");
    }
    
    if(tensor->ndim == 3){
        uint32_t shape_i = tensor->shape[0];
        uint32_t shape_j = tensor->shape[1];
        uint32_t shape_k = tensor->shape[2];
        printf("[\n");
        for(size_t i = 0; i < shape_i; i++){
            for(size_t j = 0; j < shape_j; j++){
                printf("    [ ");
                for(size_t k = 0; k < shape_k; k++){
                    float elem = tensor_get_elem(tensor, (uint32_t[]){i, j, k});
                    if(tensor->dtype == DTYPE_FP32) printf("%10.2f ", elem);
                    else if(tensor->dtype == DTYPE_INT32) printf("%d   ", (int)elem);
                }
                printf(" ]\n");
            }
            if(i <= shape_i - 2) printf("\n\n");
        }
        printf("]\n");
    }
    else if(tensor->ndim == 2){
        uint32_t shape_i = tensor->shape[0];
        uint32_t shape_j = tensor->shape[1];
        //printf("shape_i: %u, shape_j: %u ", shape_i, shape_j);
        for(size_t i = 0; i < shape_i; i++){
            printf("[ ");
            for(size_t j = 0; j < shape_j; j++){
                float elem = tensor_get_elem(tensor, (uint32_t[]){i, j});
                if(tensor->dtype == DTYPE_FP32) printf("%10.2f ", elem);
                else if(tensor->dtype == DTYPE_INT32) printf("%d    ", (int)elem);
            }
            printf(" ]\n");
        }
    }
    else if(tensor->ndim == 1){
        printf("[ ");
        for(size_t i = 0; i < 10; i++){
            float elem = tensor_get_elem(tensor, (uint32_t[]){i});
            if(tensor->dtype == DTYPE_FP32) printf("%10.2f ", elem);
            else if(tensor->dtype == DTYPE_INT32) printf("%d    ", (int)elem);
        }
        printf(" ]\n");
    }
    printf("\n");
}


// void tensor_write_fp(const Tensor *tensor, FILE *fptr){
//     fprintf(fptr, "size,%u\n", tensor->size);
//     fprintf(fptr, "ndim,%u\n", tensor->ndim);
//     fprintf(fptr, "shape,");
//     for(size_t i = 0; i < tensor->ndim; i++){
//         fprintf(fptr, "%u,", tensor->shape[i]);
//     }
//     fprintf(fptr, "\nstride,");
//     for(size_t i = 0; i < tensor->ndim; i++){
//         fprintf(fptr, "%u,", tensor->stride[i]);
//     }
//     fprintf(fptr, "\nelem_size,%u\n", tensor->elem_size);

//     if(tensor->ndim == 3){
//         for(size_t b = 0; b < tensor->shape[0]; b++){
//             for(size_t i = 0; i < tensor->shape[1]; i++){
//                 for(size_t j = 0; j < tensor->shape[2]; j++){
//                     double elem = tensor_get_elem(tensor, (uint32_t[]){b, i, j});
//                     if(tensor->dtype == DTYPE_FP64) fprintf(fptr, ",%.17g", elem);
//                     else if(tensor->dtype == DTYPE_INT32) fprintf(fptr, ",%d", (int)elem);
//                 }
//                 fprintf(fptr, "\n");
//             }
//             if(b <= tensor->shape[0] - 2)  fprintf(fptr, "\n");
//         }
//         fprintf(fptr, "\n");
//     }
//     else if(tensor->ndim == 2){
//         //printf("writing %u\n", tensor->shape[0] * tensor->shape[1]);
//         for(size_t i = 0; i < tensor->shape[0]; i++){
//             for(size_t j = 0; j < tensor->shape[1]; j++){
//                 float elem = tensor_get_elem(tensor, (uint32_t[]){i, j});

//                 fprintf(fptr, ",%.17g", elem);
//             }
//             fprintf(fptr, "\n");
//         }
//     }
//     fprintf(fptr, "\n");
// }

// void tensor_write(const Tensor *tensor, char *filename){
//     FILE *fptr = fopen(filename, "w");  // fresh file
//     fclose(fptr);

//     fptr = fopen(filename, "a");
//     if(fptr == NULL){
//         printf("Error opening file %s\n", filename);
//         exit(1);
//     }
//     fptr = fopen(filename, "a");
//     tensor_write_fp(tensor, fptr);
//     fclose(fptr);
// }

// void tensor_write(const Tensor *t, char *filename){
//     FILE *fptr = fopen(filename, "w"); // fresh file
//     fclose(fptr);

//     fptr = fopen(filename, "a");
//     if(!fptr){
//         perror("Error opening file");
//         exit(-1);
//     }
//     uint64_t json_len;
//     char json[1024] = {0};

//     uint32_t t_size =  t->size * (t->elem_size/8); 
//     json_len = snprintf(
//         json, sizeof(json),
//         "{\"%s\":{\"dtype\":\"%s\",\"shape\":[%u],\"data_offsets\":[%u,%u]}}",
//         t->name,
//         tesnor_dtype_name(t->dtype),
//         t->shape[0],
//         // t->shape[1],
//         // t->shape[2],
//         // t->shape[3],
//         0,
//         t_size
//     );

//     printf(
//         "{\"%s\":{\"dtype\":\"%s\",\"shape\":[%u],\"data_offsets\":[%u,%u]}}\n",
//         t->name,
//         tensor_dtype_name(t->dtype),
//         t->shape[0],
//         //t->shape[1],
//         // t->shape[2],
//         // t->shape[3],
//         0,
//         t_size
//     );

//     // Step 4: write file
//     fwrite(&json_len, 8, 1, fptr);         // header
//     fwrite(json, 1, json_len, fptr);       // JSON
//     fwrite(t->data, 1, t_size, fptr);   // tensor data
//     // rewind(fptr);
//     //fseek(fptr, 0, SEEK_SET);

//     // char shape_str[128] = {0};
//     // char tmp[16];

//     // for (size_t i = 0; i < t->ndim; i++) {
//     //     snprintf(tmp, sizeof(tmp), "%u", t->shape[i]);
//     //     strcat(shape_str, tmp);
//     //     if (i + 1 < t->ndim) strcat(shape_str, ",");
//     // }
//     // char json[512];
//     // int json_len = snprintf(json, sizeof(json),
//     //     "{\"%s\":{\"shape\":[%s],\"dtype\":\"%s\",\"offset\":%llu,\"length\":%llu}}",
//     //     t->name,
//     //     shape_str,
//     //     t->dtype,
//     //     t->data + sizeof(uint64_t),
//     //     t->ndim
//     // );

//     //fprintf(fptr, "uname:{%s},shape:[%u],dtype:%s\n", tensor->size, name, size, dtype_str);
//     // write JSON
//     //fwrite(json, 1, json_len, fptr);

//     // // update header length
//     // uint64_t header_len = json_len;
//     // fseek(fptr, 0, SEEK_SET);
//     // fwrite(&header_len, sizeof(header_len), 1, fptr);
//     fclose(fptr);
// }




