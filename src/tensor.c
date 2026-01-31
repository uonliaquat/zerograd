#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "../include/tensor.h"
#include "../include/utils.h"

Tensor tensor_init(void *data, const size_t * shape, const size_t ndim, const DataType dtype, const bool requires_grad, const bool random_init){

    Tensor tensor = { 0 };


    tensor.dtype = dtype;

    tensor.ndim = ndim;
    tensor.elem_size = tensor_dtype_size(dtype);
    tensor.requires_grad = requires_grad;


    // Initializing shape
    memset(tensor.shape, 0, sizeof(tensor.shape));
    memcpy(tensor.shape, shape, sizeof(size_t)*ndim);


    // Initialzing size, (total no of elements in tensor)
    tensor.size = 1;
    for(size_t i = 0; i < ndim; i++)
        tensor.size = tensor.size * tensor.shape[i];
    
    
    // Initializing stride
    tensor.stride[0] = 1;
    tensor.stride[1] = 1;
    tensor.stride[2] = 1;
    //memset(tensor.stride, 0, sizeof(tensor.stride));
    //ndim = 3
    for(int i = ndim-2; i >= 0; i--){
        tensor.stride[i] = tensor.stride[i+1] * tensor.shape[i+1];
    }

    // tensor.stride[3] = 1;
    // for(size_t i = 3; i > 0; i--){
    //     tensor.stride[i-1] = tensor.shape[i];
    // }

    // Initializing data
    tensor.data = (void*)calloc(tensor.size,  tensor.elem_size);
    if(data != NULL)
        memcpy(tensor.data, data, tensor.elem_size * tensor.size);
    else if(random_init == true){
        for(size_t i = 0; i < tensor.size; i++){
            double rand_number = rand_uniform(-1.0, 1.0);
            ((double*)tensor.data)[i] = rand_number;
        }
    }
    else{
        for(size_t i = 0; i < tensor.size; i++){
            double elem = 0;
            ((double*)tensor.data)[i] = elem;
        }
    }
    return tensor;
}

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
        input->requires_grad,
        true
    );
    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                double elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j}, elem);
            }
        }
    }
    return output_tensor;
}


Tensor tensor_repeat(Tensor *input, size_t * repeate_dims){
     Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){repeate_dims[0] == 1 ? input->shape[0] : repeate_dims[0], repeate_dims[1] == 1 ? input->shape[1] : repeate_dims[1] },
        input->ndim,
        input->dtype,
        input->requires_grad,
        true
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

void tensor_repeat_(Tensor *input, size_t * repeate_dims, Tensor *output){
    if(output->size == 0){
        *output = tensor_init(
            NULL, 
            (size_t[]){repeate_dims[0] == 1 ? input->shape[0] : repeate_dims[0], repeate_dims[1] == 1 ? input->shape[1] : repeate_dims[1] },
            input->ndim,
            input->dtype,
            input->requires_grad,
            false
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

double tensor_get_elem(const Tensor *tensor, size_t *coords){
    // size_t rows = tensor->shape[0];
    // size_t cols = tensor->shape[1];
    // assert(row < rows && col < cols);
    size_t index = 0;
    for(size_t d = 0; d < tensor->ndim; d++){
        index += coords[d] * tensor->stride[d];
    }
    if(tensor->dtype == DTYPE_DOUBLE)
        return ((double*)tensor->data)[index];
    return ((int*)tensor->data)[index];
}

void tensor_put_elem(Tensor *tensor, size_t *coords, double elem){
    size_t index = 0;
    for(size_t d = 0; d < tensor->ndim; d++){
        index += coords[d] * tensor->stride[d];
    }
    //printf("\nindex: %zu, tensor->size: %zu\n", index, tensor->size);
    assert(index < tensor->size);
    if(tensor->dtype == DTYPE_DOUBLE){
        ((double*)tensor->data)[index] = elem;
    }
    else if(tensor->dtype == DTYPE_INT){
        ((int*)tensor->data)[index] = (int)elem;
        //printf("%d\n", ((int*)tensor->data)[index]);
    }
}

Tensor tensor_transpose(const Tensor *input){
    Tensor output = tensor_init(
        NULL, 
        input->ndim == 3 ? (size_t[]){input->shape[0], input->shape[2], input->shape[1]}: (size_t[]){input->shape[1], input->shape[0]}, 
        input->ndim,
        input->dtype,
        input->requires_grad,
        true
    );


    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];

    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                double elem = tensor_get_elem(input, input->ndim ? (size_t[]){b, i, j} : (size_t[]){i, j});
                tensor_put_elem(&output, input->ndim ? (size_t[]){b, j, i}: (size_t[]){j, i}, elem);
            }
        }
    }
    return output;
}

void tensor_transpose_(const Tensor *input, Tensor *output){
    if(output->size == 0){
        *output = tensor_init(
            NULL, 
            input->ndim == 3 ? (size_t[]){input->shape[0], input->shape[2], input->shape[1]}: (size_t[]){input->shape[1], input->shape[0]}, 
            input->ndim,
            input->dtype,
            input->requires_grad,
            true
        );
    }

    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];

    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                double elem = tensor_get_elem(input, input->ndim ? (size_t[]){b, i, j} : (size_t[]){i, j});
                tensor_put_elem(output, input->ndim ? (size_t[]){b, j, i}: (size_t[]){j, i}, elem);
            }
        }
    }

}


Tensor tensor_softmax(Tensor *input, size_t dim){
    Tensor output = tensor_init(
        NULL, 
        input->shape, 
        input->ndim,
        input->dtype,
        input->requires_grad,
        true
    );
    
    size_t batch_size   = tensor_get_batch_size(input);
    size_t rows         = tensor_get_rows(input);
    size_t cols         = tensor_get_cols(input);

    if(dim == 1){
        for(size_t b = 0; b < batch_size; b++){
            for(size_t i = 0; i < rows; i++){
                double exp_sum = 0;
                for(size_t j = 0; j < cols; j++){
                    double elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});
                    exp_sum = exp_sum + expf(elem);
                }
                for(size_t j = 0; j < cols; j++){
                    double elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});

                    double new_elem = expf(elem) / exp_sum;
                    tensor_put_elem(&output, output.ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j}, new_elem);
                }
            }
        }
    }
    return output;
}
void tensor_softmax_(Tensor *input, size_t dim, Tensor *output){
    if(output->size == 0){
        *output = tensor_init(
            NULL, 
            input->shape, 
            input->ndim,
            input->dtype,
            input->requires_grad,
            true
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
                    double elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});
                    exp_sum = exp_sum + expf(elem);
                }
                for(size_t j = 0; j < cols; j++){
                    double elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});

                    double new_elem = expf(elem) / exp_sum;
                    tensor_put_elem(output, output->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j}, new_elem);
                }
            }
        }
    }
}


// Tensor tensor_scale(Tensor *input, Tensor *scale){
//     assert(scale->ndim == 2);
//     Tensor output_tensor = tensor_init(
//         NULL, 
//         input->shape, 
//         input->ndim, 
//         input->dtype, 
//         input->requires_grad, 
//         false
//     );

//     size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
//     size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
//     size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


//     for(size_t i = 0; i < batch_size; i++){
//         for(size_t j = 0; j < rows; j++){
//             double scale_factor =  tensor_get_elem(scale, (size_t[]){j, i});
//             for(size_t k = 0; k < cols; k++){
//                 double old_elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k});
//                 double new_elem = old_elem * scale_factor;
//                 tensor_put_elem(&output_tensor, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k}, new_elem);
//             }
//         }
//     }
//     return output_tensor;
// }

Tensor tensor_elementwise_scale(Tensor *input, double elem){
    Tensor output = tensor_init(
        NULL, 
        input->shape,
        input->ndim,
        input->dtype,
        input->requires_grad,
        true
    );


    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                double old_elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});
                double new_elem = old_elem * elem;
                tensor_put_elem(&output,  output.ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j}, new_elem);
            }
        }
    }
    return output;
}

void tensor_elementwise_scale_(Tensor *input, double elem, Tensor *output){
    if(output->size == 0){
        *output = tensor_init(
            NULL, 
            input->shape,
            input->ndim,
            input->dtype,
            input->requires_grad,
            true
        );
    }
    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                double old_elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});
                double new_elem = old_elem * elem;
                tensor_put_elem(output,  output->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j}, new_elem);
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
        input->requires_grad, 
        false
    );

    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            double scale_factor =  tensor_get_elem(vector, (size_t[]){i, b});
            for(size_t j = 0; j < cols; j++){
                double old_elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});
                double new_elem = old_elem * scale_factor;
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j}, new_elem);
            }
        }
    }
    return output_tensor;
}

Tensor tensor_concat(Tensor *input, size_t no_of_tensors, size_t dim){
    //this function doesn't work on batch of tensors
    if(dim == 1){
        size_t new_cols = input[0].shape[input[0].ndim -1];
        for(size_t i = 1; i < no_of_tensors; i++){
            printf("no_of_tensors: %zu\n", no_of_tensors);
            printf("input[i-1]->ndim %zu, input[i]->ndim: %zu\n", input[i-1].ndim, input[i].ndim);
            assert(input[i-1].ndim == input[i].ndim);
            assert(input[i-1].shape[0] == input[i].shape[0]);
            assert(input[i-1].shape[1] == input[i].shape[1]);
            if(input[i-1].ndim == 3){
                assert(input[i-1].shape[2] == input[2].shape[2]);
            }
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
 
        Tensor output = tensor_init(
            NULL, 
            (size_t[]){batch_size, rows, new_cols}, 
            input[0].ndim,
            input[0].dtype,
            input[0].requires_grad,
            false
        );
        for(size_t b = 0; b < batch_size; b++){
            for(size_t t = 0; t < no_of_tensors; t++){
                for(size_t i = 0; i < rows; i++){
                    for(size_t j = 0; j < cols; j++){
                        double elem = tensor_get_elem(&input[t], input[t].ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});
                        tensor_put_elem(&output,  output.ndim == 3 ? (size_t[]){b, i, j + (t * cols)}: (size_t[]){i, j + (t * cols)}, elem);
                    }
                }
            }
        }
        return output;
    }
    return (Tensor){};
}

void tensor_concat_(Tensor *input, size_t no_of_tensors, size_t dim, Tensor *output){
    //this function doesn't work on batch of tensors
    if(dim == 1){
        size_t new_cols = input[0].shape[input[0].ndim -1];
        for(size_t i = 1; i < no_of_tensors; i++){
            printf("no_of_tensors: %zu\n", no_of_tensors);
            printf("input[i-1]->ndim %zu, input[i]->ndim: %zu\n", input[i-1].ndim, input[i].ndim);
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
                (size_t[]){batch_size, rows, new_cols}, 
                input[0].ndim,
                input[0].dtype,
                input[0].requires_grad,
                false
            );
        }
        for(size_t b = 0; b < batch_size; b++){
            for(size_t t = 0; t < no_of_tensors; t++){
                for(size_t i = 0; i < rows; i++){
                    for(size_t j = 0; j < cols; j++){
                        double elem = tensor_get_elem(&input[t], input[t].ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){i, j});
                        tensor_put_elem(output,  output->ndim == 3 ? (size_t[]){b, i, j + (t * cols)}: (size_t[]){i, j + (t * cols)}, elem);
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
                input->ndim == 3 ? (size_t[]){input->shape[0], input->shape[1], cols} : (size_t[]){input->shape[0], cols}, 
                input->ndim,
                input->dtype,
                input->requires_grad,
                false
            );
            
            printf("batch_size: %zu, rows: %zu, cols: %zu\n", batch_size, rows, cols);
            for(size_t b = 0; b < batch_size; b++){
                for(size_t i = 0; i < rows; i++){
                    for(size_t j = 0; j < cols; j++){
                        double elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j + (chunk * cols)}: (size_t[]){i, j + (chunk * cols)});
                        tensor_put_elem(&output_tensor, output_tensor.ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){b, i}, elem);
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
    tensor_print(input, "input");
    assert(input->ndim <= 3);
    if(dim == 1){
        assert(input->shape[input->ndim - 1] % chunks == 0);
        //split columns
        size_t cols = input->shape[input->ndim - 1] / chunks;

        size_t batch_size = tensor_get_batch_size(input);
        size_t rows = tensor_get_rows(input);
        printf("input->shape[input->ndim - 1]: %zu, chunks: %zu\n", input->shape[input->ndim - 1], chunks);
        printf("batch_size: %zu, rows: %zu, cols: %zu\n", batch_size, rows, cols);
        //size_t cols = tensor_get_cols(input);
        //printf("chunks: %zu\n", chunks);

        for(size_t chunk = 0; chunk < chunks; chunk++){
            
            if(output[chunk].size == 0){
                output[chunk] = tensor_init(
                    NULL, 
                    input->ndim == 3 ? (size_t[]){input->shape[0], input->shape[1], cols} : (size_t[]){input->shape[0], cols}, 
                    input->ndim,
                    input->dtype,
                    input->requires_grad,
                    false
                );
            }
    
            for(size_t b = 0; b < batch_size; b++){
                for(size_t i = 0; i < rows; i++){
                    for(size_t j = 0; j < cols; j++){
                        double elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){b, i, j + (chunk * cols)}: (size_t[]){i, j + (chunk * cols)});
                        tensor_put_elem(&output[chunk], output[chunk].ndim == 3 ? (size_t[]){b, i, j}: (size_t[]){b, i}, elem);
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
        (size_t[]){tensors[0]->shape[0],  
        tensors[0]->shape[1]*len}, 
        2,
        tensors[0]->dtype,
        tensors[0]->requires_grad,
        false
    );
    size_t out_col = 0;
    for(size_t i = 0; i < len; i++){
        for(size_t j = 0; j < tensors[i]->shape[0]; j++){
            for(size_t k = 0; k < tensors[i]->shape[1]; k++){
                double elem = tensor_get_elem(tensors[i], (size_t[]){j, k});
                tensor_put_elem(&output_tensor, (size_t[]){j, k+out_col}, elem);
            }
        }
        out_col += tensors[i]->shape[1];
    }
    return output_tensor;
}

Tensor tensor_arange(const int start, const int end, const int steps){
    //returns 1d tensor only
    Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){1, (int)((end - start)/steps)}, 
        2,
        DTYPE_INT,
        false,
        false
    );
    for(size_t i = start; i <=end; i += steps){
        tensor_put_elem(&output_tensor, (size_t[]){0, (double)i}, i);
    }
    return output_tensor;
}
void tensor_arange_(const int start, const int end, const int steps, Tensor *output){
    if(output->size == 0){
      *output = tensor_init(
            NULL, 
            (size_t[]){1, (int)((end - start)/steps)}, 
            2,
            DTYPE_INT,
            false,
            false
        );
    }
    for(size_t i = start; i < end; i += steps){
        printf("%zu\n", i);
        tensor_put_elem(output, (size_t[]){0, i}, i);
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
                double elem1 = tensor_get_elem(tensor1, tensor1->ndim == 3 ?(size_t[]){batch_dim, i, k}:  (size_t[]){i, k});
                double elem2 = tensor_get_elem(tensor2, tensor2->ndim == 3 ?(size_t[]){batch_dim, k, j}: (size_t[]){k, j});
                result += elem1 * elem2;
            }
            if(output_tensor->ndim == 3) tensor_put_elem(output_tensor, (size_t[]){batch_dim, i, j}, result);
            else if(output_tensor->ndim == 2)  tensor_put_elem(output_tensor, tensor2->ndim == 3 ? (size_t[]){batch_dim, i, j}: (size_t[]){i, j}, result);
        }
    }
}


Tensor tensor_dot_product(const Tensor *input1, const Tensor *input2){
    //assert(tensor1->ndim == tensor2->ndim);
    // tensor_print(tensor1, "================== TENSOR 1 ==================");
    // tensor_print(tensor2, "================== TENSOR 2 ==================");
    size_t t1_batch_size = tensor_get_batch_size(input1);
    size_t t1_rows = tensor_get_rows(input1);
    size_t t1_cols = tensor_get_cols(input1);

    size_t t2_batch_size = tensor_get_batch_size(input2);
    size_t t2_rows = tensor_get_rows(input2);
    size_t t2_cols = tensor_get_rows(input2);

    //printf("t1_cols: %zu, t2_rows: %zu\n", t1_cols, t2_rows);
    assert(t2_batch_size == 1);
    assert(t1_cols == t2_rows);

    Tensor output =  tensor_init(
        NULL, 
        (size_t[]){t1_batch_size, t1_rows, t2_cols}, input1->ndim,
        input1->dtype,
        input1->requires_grad,
        false
    );
    for(size_t b = 0; b < t1_batch_size; b++){
        tensor_mat_mul(input1, input2, &output, b);
    }
    return output;
}

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
            (size_t[]){t1_batch_size, t1_rows, t2_cols}, input1->ndim,
            input1->dtype,
            input1->requires_grad,
            false
        );
    }
    for(size_t b = 0; b < t1_batch_size; b++){
        tensor_mat_mul(input1, input2, output, b);
    }

}





// Tensor tensor_dot_product(const Tensor *tensor1, const Tensor *tensor2){
//     assert(tensor1->shape[1] == tensor2->shape[0]);
//     Tensor output_tensor =  tensor_init(
//         NULL, 
//         (size_t[]){tensor1->shape[0],  
//         tensor2->shape[1]}, 
//         2,
//         tensor1->dtype,
//         tensor1->requires_grad,
//         true
//     );
//     size_t t1_rows = tensor1->shape[0];
//     size_t t1_cols = tensor1->shape[1];
//     size_t t2_rows = tensor2->shape[0];
//     size_t t2_cols = tensor2->shape[1];
//     size_t out_rows = output_tensor.shape[0];
//     size_t out_cols = output_tensor.shape[1];
//     for(size_t i = 0; i < out_rows; i++){
//         for(size_t j = 0; j < out_cols; j++){
//             double result = 0;
//             for(size_t k = 0; k < t1_cols; k++){
//                 double elem1 = tensor_get_elem(tensor1, (size_t[]){i, k});
//                 double elem2 = tensor_get_elem(tensor2, (size_t[]){k, j});
//                 result += elem1 * elem2;
//             }
//             tensor_put_elem(&output_tensor, i, j, result);
//         }
//     }
//     return output_tensor;
// }

Tensor tensor_add(Tensor *input1, Tensor *input2){
    Tensor output = tensor_init(
        input1->data, 
        input1->shape, 
        input1->ndim, 
        input1->dtype, 
        input1->requires_grad, 
        false
    );
    for(size_t i = 0; i < output.size; i++){
        ((double*)output.data)[i] = ((double*)output.data)[i] + ((double*)input2->data)[i];
    }
    return output;
}

void tensor_add_(Tensor *input1, Tensor *input2, Tensor *output){
    assert(input1->dtype == input2->dtype);
    if(output->size == 0){
        *output = tensor_init(
            input1->data, 
            input1->shape, 
            input1->ndim, 
            input1->dtype, 
            input1->requires_grad, 
            false
        );
    }
    memcpy(output->data, input1->data, input1->size * tensor_dtype_size(input1->dtype));
    for(size_t i = 0; i < output->size; i++){
        ((double*)output->data)[i] = ((double*)output->data)[i] + ((double*)input2->data)[i];
    }
}

Tensor tensor_elementwise_add(Tensor *tensor, double val){

    Tensor outout_tensor = tensor_init(
        tensor->data, 
        tensor->shape, 
        tensor->ndim, 
        tensor->dtype, 
        tensor->requires_grad, 
        false
    );

    size_t batch_size = tensor->ndim == 3 ? tensor->shape[0]: 1;
    size_t rows = tensor->ndim == 3 ? tensor->shape[1]: tensor->shape[0];
    size_t cols = tensor->ndim == 3 ? tensor->shape[2]: tensor->shape[1];

    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(tensor, tensor->ndim ==3 ? (size_t[]){i, j, k}:  (size_t[]){j, k});
                tensor_put_elem(tensor, tensor->ndim ==3 ? (size_t[]){i, j, k}:  (size_t[]){j, k}, val * elem);
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
        input->requires_grad, 
        false
    );

    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double scale_factor =  tensor_get_elem(vector, (size_t[]){j, i});
            for(size_t k = 0; k < cols; k++){
                double old_elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k});
                double new_elem = old_elem + scale_factor;
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k}, new_elem);
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
        input->requires_grad, 
        false
    );


    // Tensor new_tensor = tensor_init(NULL, shape, ndim, dtype, false, false);
    for(size_t i = 0; i < input->shape[0]; i++){
        for(size_t j = 0; j < input->shape[1]; j++){
            if(j > i) tensor_put_elem(input, (size_t[]){i,j}, elem);
            // else tensor_put_elem(&new_tensor, (size_t[]){i,j}, 0);
        }
    }
    return output;
}

void tensor_tril_(Tensor *input, double elem, Tensor *output){
    if(output->size == 0){

    }
}

void tensor_masked_fill(Tensor *tensor, double mask, double fill){
    assert(tensor->ndim == 2);
    for(size_t i = 0; i < tensor->shape[0]; i++){
        for(size_t j = 0; j < tensor->shape[1]; j++){
            double elem = tensor_get_elem(tensor, (size_t[]){i, j});
            if(elem == mask){
                tensor_put_elem(tensor, (size_t[]){i, j}, fill);
            }
        }
    }
}



void tensor_copy_row_data(Tensor *dest_tensor, size_t batch_id, size_t row_id, Tensor *src_tensor, size_t src_row, size_t no_of_items){
    size_t dest_index = batch_id * dest_tensor->stride[0] + row_id * dest_tensor->stride[1];
    size_t src_index =  src_row * src_tensor->stride[0];
    //printf("batch_id %zu, dest_tensor->stride[0]: %zu, row_id: %zu,  dest_tensor->stride[1]: %zu\n", batch_id, dest_tensor->stride[0], row_id, dest_tensor->stride[1]);
    // printf("dest index %zu, ", dest_index);
    // printf("src_index: %zu\n", src_index);
    void *dest, *src;
    if(dest_tensor->dtype == DTYPE_DOUBLE){
        dest = &((double*)dest_tensor->data)[dest_index];
        src =  &((double*)src_tensor->data)[src_index];
    }
    else if(dest_tensor->dtype == DTYPE_INT){
        dest = &((int*)dest_tensor->data)[dest_index];
        src =  &((int*)src_tensor->data)[src_index];
    }
    memcpy(dest, src, no_of_items * src_tensor->elem_size);
}

Tensor tensor_mean_var(Tensor *x){

    Tensor output_tensor = tensor_init(
        NULL, 
        x->ndim ==3 ? (size_t[]){x->shape[0], x->shape[1], 2}:  (size_t[]){x->shape[0], 2}, 
        x->ndim, 
        x->dtype, 
        x->requires_grad, 
        false
    );
    size_t batch_size = x->ndim == 3 ? x->shape[0]: 1;
    size_t rows = x->ndim == 3 ? x->shape[1]: x->shape[0];
    size_t cols = x->ndim == 3 ? x->shape[2]: x->shape[1];
    //printf("batch_size: %zu, rows: %zu, cols: %zu, x->ndim: %zu\n", batch_size, rows, cols, x->ndim);
    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double mean = 0;
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(x, x->ndim ==3 ? (size_t[]){i, j, k}:  (size_t[]){j, k});
                mean += elem;
            }
            mean = mean / cols;
            printf("%f\n", mean);

            double variance = 0;
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(x, x->ndim ==3 ? (size_t[]){i, j, k}:  (size_t[]){j, k});
                double squared_deviation = pow(elem - mean, 2);
                //double sqrt_squared_deviation = sqrt(squared_deviation);
                variance += squared_deviation;
                //printf("elem: %.3f, mean: %.3f, elem-mean: %.3f, squared_deviation: %.3f, sqrt_squared_deviation: %.3f\n", elem, mean, elem - mean, squared_deviation, sqrt_squared_deviation);
            }
            variance = variance / cols; 
            tensor_put_elem(&output_tensor, x->ndim ==3 ? (size_t[]){i, j, 0}:  (size_t[]){j, 0}, mean);
            tensor_put_elem(&output_tensor, x->ndim ==3 ? (size_t[]){i, j, 1}:  (size_t[]){j, 1}, variance);
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
        x->requires_grad, 
        false
    );

    size_t batch_size = x->ndim == 3 ? x->shape[0]: 1;
    size_t rows = x->ndim == 3 ? x->shape[1]: x->shape[0];
    size_t cols = x->ndim == 3 ? x->shape[2]: x->shape[1];
    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double mean = tensor_get_elem(mean_var_tensor, x->ndim == 3 ? (size_t[]){i, j, 0}: (size_t[]){j, 0});
            double var = tensor_get_elem(mean_var_tensor, x->ndim == 3 ? (size_t[]){i, j, 1}: (size_t[]){j, 1});
            //printf("mean: %.2f, var: %.2f\n", mean, var);
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(x, x->ndim == 3? (size_t[]){i, j, k}: (size_t[]){j, k});
                double norm_elem = (elem - mean) / sqrt(var + eps);
                tensor_put_elem(&output_tensor, x->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k}, norm_elem);
            }
        }
    }
    return output_tensor;
}

void tensor_print(const Tensor *tensor, const char *heading){
    printf("\n============== %s ==================\n", heading);
    printf("size:           %zu\n", tensor->size);
    printf("ndim:           %zu\n", tensor->ndim);
    
    printf("shape:          ( ");
    for(size_t i = 0; i < tensor->ndim; i++){
        printf("%zu, ", tensor->shape[i]);
    }
    printf(" )\n");
    printf("stride:         ( ");
    for(size_t i = 0; i < tensor->ndim; i++){
        printf("%zu, ", tensor->stride[i]);
    }
    printf(" )\n");
    printf("elem_size:      %zu\n", tensor->elem_size);
    printf("requires_grad:  %s\n", tensor->requires_grad ? "true" : "false");
    printf("data:\n");
    
    if(tensor->ndim == 3){
        printf("[\n");
        for(size_t b = 0; b < tensor->shape[0]; b++){
            for(size_t i = 0; i < tensor->shape[1]; i++){
                printf("    [ ");
                for(size_t j = 0; j < tensor->shape[2]; j++){
                    double elem = tensor_get_elem(tensor, (size_t[]){b, i, j});
                    if(tensor->dtype == DTYPE_DOUBLE) printf("%10.2f ", elem);
                    else if(tensor->dtype == DTYPE_INT) printf("%d   ", (int)elem);
                }
                printf(" ]\n");
            }
            if(b <= tensor->shape[0] - 2) printf("\n\n");
        }
        printf("]\n");
    }
    else if(tensor->ndim == 2){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            printf("[ ");
            for(size_t j = 0; j < tensor->shape[1]; j++){
                double elem = tensor_get_elem(tensor, (size_t[]){i, j});
                if(tensor->dtype == DTYPE_DOUBLE) printf("%10.2f ", elem);
                else if(tensor->dtype == DTYPE_INT) printf("%d    ", (int)elem);
            }
            printf(" ]\n");
        }
    }
    printf("\n");
}


void tensor_write_fp(const Tensor *tensor, FILE *fptr){
    fprintf(fptr, "size,%zu\n", tensor->size);
    fprintf(fptr, "ndim,%zu\n", tensor->ndim);
    fprintf(fptr, "shape,");
    for(size_t i = 0; i < tensor->ndim; i++){
        fprintf(fptr, "%zu,", tensor->shape[i]);
    }
    fprintf(fptr, "\nstride,");
    for(size_t i = 0; i < tensor->ndim; i++){
        fprintf(fptr, "%zu,", tensor->stride[i]);
    }
    fprintf(fptr, "\nelem_size,%zu\n", tensor->elem_size);
    fprintf(fptr, "requires_grad,%d\n\n", tensor->requires_grad);

    if(tensor->ndim == 3){
        for(size_t b = 0; b < tensor->shape[0]; b++){
            for(size_t i = 0; i < tensor->shape[1]; i++){
                for(size_t j = 0; j < tensor->shape[2]; j++){
                    double elem = tensor_get_elem(tensor, (size_t[]){b, i, j});
                    if(tensor->dtype == DTYPE_DOUBLE) fprintf(fptr, ",%.17g", elem);
                    else if(tensor->dtype == DTYPE_INT) fprintf(fptr, ",%d", (int)elem);
                }
                fprintf(fptr, "\n");
            }
            if(b <= tensor->shape[0] - 2)  fprintf(fptr, "\n");
        }
        fprintf(fptr, "\n");
    }
    else if(tensor->ndim == 2){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            for(size_t j = 0; j < tensor->shape[1]; j++){
                double elem = tensor_get_elem(tensor, (size_t[]){i, j});
                fprintf(fptr, ",%.17g", elem);
            }
            fprintf(fptr, "\n");
        }
    }
    fprintf(fptr, "\n");
}

void tensor_write(const Tensor *tensor, char *filename){
    FILE *fptr = fopen(filename, "w");  // fresh file
    fclose(fptr);

    fptr = fopen(filename, "a");
    if(fptr == NULL){
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    fptr = fopen(filename, "a");
    tensor_write_fp(tensor, fptr);
}


size_t tensor_dtype_size(const DataType dtype){
    switch (dtype){
        case DTYPE_DOUBLE: return sizeof(double);
        case DTYPE_INT: return sizeof(int);
    }
}

