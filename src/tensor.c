#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>

#include "../include/tensor.h"
#include "../include/utils.h"



static inline void shape_init(Tensor *tensor, const uint32_t *shape){
    // Initializing shape
    memset(tensor->shape, 0, TENSOR_MAX_SHAPE_DIM);
    memcpy(tensor->shape, shape, tensor->ndim * sizeof(uint32_t));
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

static inline void name_init(Tensor *tensor, const char *name){
    if(name != NULL){
        memset(tensor->name, 0, TENSOR_MAX_LEN_NAME);
        memcpy(tensor->name, name, strlen(name));
    }
}

static inline void data_init(Tensor *tensor, void *data){
    // Initializing data
    tensor->data = calloc(tensor->size,  tensor->elem_size);
    if(data && tensor->data) memcpy(tensor->data, data, tensor->size *  tensor->elem_size);
    else memset(tensor->data, 0, tensor->size);
    // printf("Sanity Check\n");
    // for(size_t i = 0; i < 10; i++){
    //     printf("%f\n", ((float*)tensor->data)[i]);
    // }
}

static inline void data_rand_init(Tensor *tensor){
    tensor->data = (void*)calloc(tensor->size,   tensor->elem_size);
    for(size_t i = 0; i < tensor->size; i++){
        float rand_number = rand_uniform(-1.0, 1.0);
        ((float*)tensor->data)[i] = rand_number;
    }
}

static inline void data_file_init(Tensor *tensor, FILE *fptr, const uint32_t *offset){
    tensor->data = (void*)calloc(tensor->size,   tensor->elem_size);
    for(size_t i = 0; i < tensor->size; i++){
        float elem = 0;
        ((float*)tensor->data)[i] = 0;
    }
}

Tensor tensor_init(void *data, const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name){

    Tensor tensor;
    tensor_reset(&tensor, name);

    tensor.ndim = ndim;
    tensor.dtype = dtype;
    tensor.elem_size = tensor_dtype_size(dtype);
    
    shape_init(&tensor, shape);
    size_init(&tensor);
    stride_init(&tensor);
    //name_init(&tensor, name);
    data_init(&tensor, data);
    return tensor;
}

void tensor_init_(Tensor *tensor, void *data, const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name){
    tensor_reset(tensor, name);

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->elem_size = tensor_dtype_size(dtype);

    shape_init(tensor, shape);
    size_init(tensor);
    stride_init(tensor);
    //name_init(tensor, name);
    data_init(tensor, data);
}

void tensor_reset(Tensor *tensor, const char *name){
    if(tensor != NULL){
        tensor->dtype = 0;
        tensor->ndim = 0;
        tensor->elem_size = 0;
        tensor->size = 0;
        tensor->data = NULL;
        name_init(tensor, name);
        memset(tensor->shape, 0, sizeof(tensor->shape));
        memset(tensor->stride, 0, sizeof(tensor->stride));
    }
}

Tensor tensor_rand_init(const uint32_t * shape, const uint8_t ndim, const DataType dtype, char *name){

    Tensor tensor;
    tensor_reset(&tensor, name);

    tensor.ndim = ndim;
    tensor.dtype = dtype;
    tensor.elem_size = tensor_dtype_size(dtype);

    shape_init(&tensor, shape);
    size_init(&tensor);
    stride_init(&tensor);
    //name_init(&tensor, name);
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
                float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, elem);
            }
        }
    }
    return output_tensor;
}


void tensor_copy_(Tensor *input, Tensor *output)
{   
    if(output->size == 0){
        tensor_init_(
            output,
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
                float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                tensor_put_elem(output, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, elem);
            }
        }
    }
}


Tensor tensor_repeat(Tensor *input, uint8_t * repeate_dims){
     Tensor output_tensor = tensor_init(
        NULL, 
        (uint32_t[]){repeate_dims[0] == 1 ? input->shape[0] : repeate_dims[0], repeate_dims[1] == 1 ? input->shape[1] : repeate_dims[1] },
        2,
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
            tensor_copy_row_data(&output_tensor, i, 0, input, 0, output_tensor.size);
        }
    }
    return output_tensor;
}

void tensor_repeat_(Tensor *input, uint8_t * repeate_dims, Tensor *output){
    if(output->size == 0){
        tensor_init_(
            output,
            NULL, 
            (uint32_t[]){repeate_dims[0] == 1 ? input->shape[0] : repeate_dims[0], repeate_dims[1] == 1 ? input->shape[1] : repeate_dims[1], repeate_dims[2] == 1 ? input->shape[2] : repeate_dims[2]  },
            3,
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
            printf("input->size: %u, input->elem_size: %u\n", input->size, input->elem_size);
            tensor_copy_row_data(output, i, 0, input, 0, input->size);
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
    if(tensor->dtype == DTYPE_FP32){
        return ((float*)tensor->data)[index];
    }
    else if(tensor->dtype == DTYPE_INT32){
        return ((int*)tensor->data)[index];
    }
    return 0;
}

void tensor_put_elem(Tensor *tensor, uint32_t *coords, float elem){
    size_t index = 0;
    //printf("tensor->ndim: %u\n", tensor->ndim);
    for(size_t d = 0; d < tensor->ndim; d++){
        //printf("cords: %u, stride: %u\n", coords[d],  tensor->stride[d]);
        index += coords[d] * tensor->stride[d];
    }
    //printf("\nindex: %zu, tensor->size: %u\n\n\n", index, tensor->size);
    assert(index < tensor->size);
    if(tensor->dtype == DTYPE_FP32){
        ((float*)tensor->data)[index] = elem;
    }
    else if(tensor->dtype == DTYPE_INT32){
        ((int*)tensor->data)[index] = (int)elem;
        //printf("%d\n", ((int*)tensor->data)[index]);
    }
}

Tensor tensor_transpose(const Tensor *input){
    Tensor output = tensor_init(
        NULL, 
        input->ndim == 3 ? (uint32_t[]){input->shape[0], input->shape[2], input->shape[1]}: (uint32_t[]){input->shape[1], input->shape[0]}, 
        input->ndim,
        input->dtype,
        "tensor_transpose.ouput"
    );


    size_t batch_size = tensor_get_batch_size(input);
    size_t rows = tensor_get_rows(input);
    size_t cols = tensor_get_cols(input);

    // printf("batch_size: %zu\n", batch_size);
    // printf("rows: %zu\n", rows);
    // printf("cols: %zu\n", cols);

    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < cols; j++){
                //printf("i: %zu, j: %zu\n", i, j);
                float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j} : (uint32_t[]){i, j});
                tensor_put_elem(&output, output.ndim == 3 ? (uint32_t[]){b, j, i}: (uint32_t[]){j, i}, elem);
                //tensor_print(&output, "inside");
            }
        }
    }
    return output;
}

void tensor_transpose_(const Tensor *input, Tensor *output){
    if(output->size == 0){
        tensor_init_(
            output,
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
                float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j} : (uint32_t[]){i, j});
                tensor_put_elem(output, input->ndim == 3 ? (uint32_t[]){b, j, i}: (uint32_t[]){j, i}, elem);
            }
        }
    }

}


// Tensor tensor_softmax(Tensor *input, size_t dim){
//     Tensor output = tensor_init(
//         NULL, 
//         input->shape, 
//         input->dtype,
//         true
//     );
    
//     size_t batch_size   = tensor_get_batch_size(input);
//     size_t rows         = tensor_get_rows(input);
//     size_t cols         = tensor_get_cols(input);

//     if(dim == 1){
//         for(size_t b = 0; b < batch_size; b++){
//             for(size_t i = 0; i < rows; i++){
//                 float exp_sum = 0;
//                 for(size_t j = 0; j < cols; j++){
//                     float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
//                     exp_sum = exp_sum + expf(elem);
//                 }
//                 for(size_t j = 0; j < cols; j++){
//                     float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});

//                     float new_elem = expf(elem) / exp_sum;
//                     tensor_put_elem(&output, output.ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
//                 }
//             }
//         }
//     }
//     return output;
// }
void tensor_softmax_(Tensor *input, size_t dim, Tensor *output){
    if(output->size == 0){
        tensor_init_(
            output,
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
    const float EPS = 1e-12f;
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < rows; i++) {
            float max_elem = -FLT_MAX;
            
            // Find max
            for (size_t j = 0; j < cols; j++) {
                float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b,i,j} : (uint32_t[]){i,j});
                if (elem > max_elem) max_elem = elem;
            }
            
            // Compute exp_sum
            float exp_sum = 0.0f;
            for (size_t j = 0; j < cols; j++) {
                float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b,i,j} : (uint32_t[]){i,j});
                float e = expf(elem - max_elem);
                if (!isfinite(e)) e = 0.0f;
                exp_sum += e;
            }
            
            exp_sum = fmaxf(exp_sum, EPS);  // Prevent division by zero
            
            // Normalize
            for (size_t j = 0; j < cols; j++) {
                float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b,i,j} : (uint32_t[]){i,j});
                float e = expf(elem - max_elem);
                if (!isfinite(e)) e = 0.0f;  // Apply guard here too
                float new_elem = e / exp_sum;
                tensor_put_elem(output, output->ndim == 3 ? (uint32_t[]){b,i,j} : (uint32_t[]){i,j}, new_elem);
            }
        }
    }
}

void tensor_gelu_(Tensor *input, Tensor *output){
    if(output->size == 0){
        tensor_init_(
            output,
            NULL,
            input->shape,
            input->ndim,
            input->dtype,
            NULL
        );
    }

    for(size_t i = 0; i < input->size; i++){
        float x = ((float*)input->data)[i];
        float gelu = x * 0.5 * (1.0 + erf(x / sqrt(2.0)));
        ((float*)output->data)[i] = gelu;
    }
}


// Tensor tensor_scale(Tensor *input, Tensor *scale){
//     assert(scale->ndim == 2);
//     Tensor output_tensor = tensor_init(
//         NULL, 
//         input->shape, 
//         input->dtype, 
//         false
//     );

//     size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
//     size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
//     size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


//     for(size_t i = 0; i < batch_size; i++){
//         for(size_t j = 0; j < rows; j++){
//             float scale_factor =  tensor_get_elem(scale, (uint32_t[]){j, i});
//             for(size_t k = 0; k < cols; k++){
//                 float old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k});
//                 float new_elem = old_elem * scale_factor;
//                 tensor_put_elem(&output_tensor, input->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k}, new_elem);
//             }
//         }
//     }
//     return output_tensor;
// }

// Tensor tensor_elementwise_scale(Tensor *input, float elem){
//     Tensor output = tensor_init(
//         NULL, 
//         input->shape,
//         input->dtype,
//         NULL
//     );


//     size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
//     size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
//     size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
//     for(size_t b = 0; b < batch_size; b++){
//         for(size_t i = 0; i < rows; i++){
//             for(size_t j = 0; j < cols; j++){
//                 float old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
//                 float new_elem = old_elem * elem;
//                 tensor_put_elem(&output,  output.ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
//             }
//         }
//     }
//     return output;
// }

void tensor_elementwise_scale_(Tensor *input, float elem, Tensor *output){
    if(output->size == 0){
        tensor_init_(
            output,
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
                float old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                float new_elem = old_elem * elem;
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
            float scale_factor =  tensor_get_elem(vector, (uint32_t[]){i, b});
            for(size_t j = 0; j < cols; j++){
                float old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                float new_elem = old_elem * scale_factor;
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
            }
        }
    }
    return output_tensor;
}

void tensor_vector_scale_(Tensor *input, Tensor *vector, Tensor *output){
    if(output->size == 0){
        tensor_init_(
            output,
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
            //printf("scale: factor: %.f\n", scale_factor);
            //float scale_factor =  tensor_get_elem(vector, vector->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
            for(size_t j = 0; j < cols; j++){
                float scale_factor =  tensor_get_elem(vector, (uint32_t[]){j});
                float old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                float new_elem = old_elem * scale_factor;
                tensor_put_elem(output, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
            }
        }
    }
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
//                         float elem = tensor_get_elem(&input[t], input[t].ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
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
            tensor_init_(
                output,
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
                        float elem = tensor_get_elem(&input[t], input[t].ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
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

        //printf("chunks: %zu\n", chunks);
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
            
            //printf("batch_size: %zu, rows: %zu, cols: %zu\n", batch_size, rows, cols);
            for(size_t b = 0; b < batch_size; b++){
                for(size_t i = 0; i < rows; i++){
                    for(size_t j = 0; j < cols; j++){
                        float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j + (chunk * cols)}: (uint32_t[]){i, j + (chunk * cols)});
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
    //printf("\n\n\n\n\n ******************* CHUNK ****************\n");
    //tensor_print(input, "input");
    assert(input->ndim <= 3);
    if(dim == 1){
        assert(input->shape[input->ndim - 1] % chunks == 0);
        //split columns
        size_t batch_size   = tensor_get_batch_size(input);
        size_t rows         = tensor_get_rows(input);
        size_t cols         = input->shape[input->ndim - 1] / chunks;
        // printf("input->shape[input->ndim - 1]: %u, chunks: %zu\n", input->shape[input->ndim - 1], chunks);
        // printf("batch_size: %zu, rows: %zu, cols: %zu\n", batch_size, rows, cols);
        //size_t cols = tensor_get_cols(input);
        // printf("chunks: %zu\n", chunks);
        for(size_t chunk = 0; chunk < chunks; chunk++){
            if(output[chunk].size == 0){
                tensor_init_(
                    &output[chunk],
                    NULL, 
                    input->ndim == 3 ? (uint32_t[]){input->shape[0], input->shape[1], cols} : (uint32_t[]){input->shape[0], cols}, 
                    input->ndim,
                    input->dtype,
                    NULL
                );
                //tensor_print(&output[chunk], " output[chunk] (initialized)");
                
            }
            for(size_t b = 0; b < batch_size; b++){
                for(size_t i = 0; i < rows; i++){
                    for(size_t j = 0; j < cols; j++){
                        float elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j + (chunk * cols)}: (uint32_t[]){i, j + (chunk * cols)});
                        tensor_put_elem(&output[chunk], output[chunk].ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){b, i}, elem);
                    }
                }
            }
            //tensor_print(&output[chunk], " output[chunk]");
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
                float elem = tensor_get_elem(tensors[i], (uint32_t[]){j, k});
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
//         tensor_put_elem(&output_tensor, (uint32_t[]){0, (float)i}, i);
//     }
//     return output_tensor;
// }
void tensor_arange_(const int start, const int end, const int steps, Tensor *output){
    if(output->size == 0){
        tensor_init_(
            output,
            NULL, 
            (uint32_t[]){1, (int)((end - start)/steps)}, 
            2,
            DTYPE_INT32,
            NULL
        );
    }
    for(size_t i = start; i < end; i += steps){
        tensor_put_elem(output, (uint32_t[]){0, i}, i);
    }
}


// void tensor_mat_mul(const Tensor *tensor1, const Tensor *tensor2, Tensor *output_tensor, size_t batch_dim){

//     size_t t1_batch_size = tensor1->ndim == 3 ? tensor1->shape[0]: 1;
//     size_t t1_rows = tensor1->ndim == 3 ? tensor1->shape[1]: tensor1->shape[0];
//     size_t t1_cols = tensor1->ndim == 3 ? tensor1->shape[2]: tensor1->shape[1];

//     size_t t2_batch_size = tensor2->ndim == 3 ? tensor2->shape[0]: 1;
//     size_t t2_rows = tensor2->ndim == 3 ? tensor2->shape[1]: tensor2->shape[0];
//     size_t t2_cols = tensor2->ndim == 3 ? tensor2->shape[2]: tensor2->shape[1];

//     //size_t t2_batch_size = tensor2->ndim == 3 ? tensor2->shape[0]: 1;
//     size_t out_rows = output_tensor->ndim == 3 ? output_tensor->shape[1]: output_tensor->shape[0];
//     size_t out_cols = output_tensor->ndim == 3 ? output_tensor->shape[2]: output_tensor->shape[1];
    
//     for(size_t i = 0; i < out_rows; i++){
//         for(size_t j = 0; j < out_cols; j++){
//             double result = 0;
//             for(size_t k = 0; k < t1_cols; k++){
//                 float elem1 = tensor_get_elem(tensor1, tensor1->ndim == 3 ?(uint32_t[]){batch_dim, i, k}: (uint32_t[]){i, k});
//                 float elem2 = tensor_get_elem(tensor2, tensor2->ndim == 3 ?(uint32_t[]){batch_dim, k, j}: (uint32_t[]){k, j});
//                 result += (elem1 * elem2);
//             }
//             if(output_tensor->ndim == 3) tensor_put_elem(output_tensor, (uint32_t[]){batch_dim, i, j}, result);
//             else if(output_tensor->ndim == 2)  tensor_put_elem(output_tensor, tensor2->ndim == 3 ? (uint32_t[]){batch_dim, i, j}: (uint32_t[]){i, j}, result);
//         }
//     }
// }

// Remove these lines from your tensor.c file:
// #define ACCELERATE_NEW_LAPACK
// #define ACCELERATE_LAPACK_ILP64

#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

void tensor_mat_mul(const Tensor *tensor1, const Tensor *tensor2, Tensor *output_tensor, size_t batch_dim) {
    size_t M = tensor1->ndim == 3 ? tensor1->shape[1] : tensor1->shape[0];
    size_t K = tensor1->ndim == 3 ? tensor1->shape[2] : tensor1->shape[1];
    size_t N = tensor2->ndim == 3 ? tensor2->shape[2] : tensor2->shape[1];
    
    size_t a_batch_offset = tensor1->ndim == 3 ? batch_dim * tensor1->stride[0] : 0;
    size_t b_batch_offset = tensor2->ndim == 3 ? batch_dim * tensor2->stride[0] : 0;
    size_t c_batch_offset = output_tensor->ndim == 3 ? batch_dim * output_tensor->stride[0] : 0;
    
    size_t a_row_stride = tensor1->ndim == 3 ? tensor1->stride[1] : tensor1->stride[0];
    size_t a_col_stride = tensor1->ndim == 3 ? tensor1->stride[2] : tensor1->stride[1];
    size_t b_row_stride = tensor2->ndim == 3 ? tensor2->stride[1] : tensor2->stride[0];
    size_t b_col_stride = tensor2->ndim == 3 ? tensor2->stride[2] : tensor2->stride[1];
    size_t c_row_stride = output_tensor->ndim == 3 ? output_tensor->stride[1] : output_tensor->stride[0];
    size_t c_col_stride = output_tensor->ndim == 3 ? output_tensor->stride[2] : output_tensor->stride[1];
    
    const float *A = (const float*)tensor1->data + a_batch_offset;
    const float *B = (const float*)tensor2->data + b_batch_offset;
    float *C = (float*)output_tensor->data + c_batch_offset;
    
    // Check if data is contiguous (can use Accelerate)
    if (a_col_stride == 1 && b_col_stride == 1 && c_col_stride == 1) {
        // Fast path: Use Accelerate framework
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            (int)M,                    // Rows of A
            (int)N,                    // Cols of B
            (int)K,                    // Cols of A / Rows of B
            1.0f,                      // alpha
            A,                         // Matrix A
            (int)a_row_stride,         // Leading dimension of A
            B,                         // Matrix B
            (int)b_row_stride,         // Leading dimension of B
            0.0f,                      // beta
            C,                         // Matrix C (output)
            (int)c_row_stride          // Leading dimension of C
        );
    } else {
        // Fallback: Custom NEON implementation for strided data
        // Initialize output
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                C[i * c_row_stride + j * c_col_stride] = 0.0f;
            }
        }
        
        // ikj ordering with NEON
        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                float a_val = A[i * a_row_stride + k * a_col_stride];
                float32x4_t a_vec = vdupq_n_f32(a_val);
                
                size_t j = 0;
                // SIMD loop (process 4 elements at once)
                for (; j + 3 < N; j += 4) {
                    size_t c_idx = i * c_row_stride + j * c_col_stride;
                    size_t b_idx = k * b_row_stride + j * b_col_stride;
                    
                    // Load B elements (may be strided)
                    float32x4_t b_vec = {
                        B[b_idx],
                        B[b_idx + b_col_stride],
                        B[b_idx + 2 * b_col_stride],
                        B[b_idx + 3 * b_col_stride]
                    };
                    
                    // Load C elements (may be strided)
                    float32x4_t c_vec = {
                        C[c_idx],
                        C[c_idx + c_col_stride],
                        C[c_idx + 2 * c_col_stride],
                        C[c_idx + 3 * c_col_stride]
                    };
                    
                    // FMA: c = c + a * b
                    c_vec = vfmaq_f32(c_vec, a_vec, b_vec);
                    
                    // Store C elements back (may be strided)
                    C[c_idx] = vgetq_lane_f32(c_vec, 0);
                    C[c_idx + c_col_stride] = vgetq_lane_f32(c_vec, 1);
                    C[c_idx + 2 * c_col_stride] = vgetq_lane_f32(c_vec, 2);
                    C[c_idx + 3 * c_col_stride] = vgetq_lane_f32(c_vec, 3);
                }
                
                // Handle remainder
                for (; j < N; j++) {
                    C[i * c_row_stride + j * c_col_stride] += a_val * B[k * b_row_stride + j * b_col_stride];
                }
            }
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
//         (uint32_t[]){t1_batch_size, t1_rows, t2_cols},
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
    // tensor_print(input1);
    // tensor_print(input2);
    // tensor_print(output);
    size_t t1_batch_size = tensor_get_batch_size(input1);
    size_t t1_rows = tensor_get_rows(input1);
    size_t t1_cols = tensor_get_cols(input1);

    size_t t2_batch_size = tensor_get_batch_size(input2);
    size_t t2_rows = tensor_get_rows(input2);
    size_t t2_cols = tensor_get_cols(input2);

    // assert(t2_batch_size == 1);
    // printf("t1_batch_size: %zu, t2_batch_size: %zu\n", t1_batch_size, t2_batch_size);
    // printf("t1_rows: %zu, t1_cols: %zu, t2_rows: %zu, t2_cols: %zu\n", t1_rows, t1_cols, t2_rows, t2_cols);
    assert(t1_cols == t2_rows);

    
    if(output->size == 0){
        tensor_init_(
            output,
            NULL, 
            (uint32_t[]){t1_batch_size, t1_rows, t2_cols}, 
            3,
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
// //             float result = 0;
// //             for(size_t k = 0; k < t1_cols; k++){
// //                 float elem1 = tensor_get_elem(tensor1, (uint32_t[]){i, k});
// //                 float elem2 = tensor_get_elem(tensor2, (uint32_t[]){k, j});
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
//         input1->dtype, 
//         NULL
//     );
//     for(size_t i = 0; i < output.size; i++){
//         ((float*)output.data)[i] = ((float*)output.data)[i] + ((float*)input2->data)[i];
//     }
//     return output;
// }

void tensor_add_(Tensor *input1, Tensor *input2, Tensor *output){
    assert(input1->dtype == input2->dtype);
    if(output->size == 0){
        tensor_init_(
            output,
            NULL, 
            input1->shape, 
            input1->ndim,
            input1->dtype,
            NULL
        );
    }
    tensor_copy_(input1, output);
    for(size_t i = 0; i < output->size; i++){
        double elem = ((float*)output->data)[i] + ((float*)input2->data)[i];
        ((float*)output->data)[i] = elem;
    }
}

Tensor tensor_elementwise_add(Tensor *tensor, float val){

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
                float elem = tensor_get_elem(tensor, tensor->ndim ==3 ? (uint32_t[]){i, j, k}:  (uint32_t[]){j, k});
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
            float scale_factor =  tensor_get_elem(vector, (uint32_t[]){j, i});
            for(size_t k = 0; k < cols; k++){
                float old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k});
                float new_elem = old_elem + scale_factor;
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k}, new_elem);
            }
        }
    }
    return output_tensor;
}


void tensor_vector_add_(Tensor *input, Tensor *vector, Tensor *output){
    if(output->size == 0){
        tensor_init_(
            output,
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
            // float shift_factor =  tensor_get_elem(vector, (uint32_t[]){i});
            for(size_t j = 0; j < cols; j++){
                float shift_factor =  tensor_get_elem(vector, (uint32_t[]){j});
                float old_elem = tensor_get_elem(input, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                float new_elem = old_elem + shift_factor;
                tensor_put_elem(output, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, new_elem);
            }
        }
    }
}

Tensor tensor_tril(Tensor *input, float elem){
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

void tensor_tril_(Tensor *tensor, float elem){
    size_t batch_size = tensor->ndim == 3 ? tensor->shape[0]: 1;
    size_t rows = tensor->ndim == 3 ? tensor->shape[1]: tensor->shape[0];
    size_t cols = tensor->ndim == 3 ? tensor->shape[2]: tensor->shape[1];

    for(size_t b = 0; b < batch_size; b++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = i+1; j < cols; j++){
                tensor_put_elem(tensor, tensor->ndim ==3 ? (uint32_t[]){b, i, j}:  (uint32_t[]){i, j}, elem);
            }
        }
    }
}

void tensor_masked_fill(Tensor *tensor, float mask, float fill){
    assert(tensor->ndim == 2);
    for(size_t i = 0; i < tensor->shape[0]; i++){
        for(size_t j = 0; j < tensor->shape[1]; j++){
            float elem = tensor_get_elem(tensor, (uint32_t[]){i, j});
            if(elem == mask){
                tensor_put_elem(tensor, (uint32_t[]){i, j}, fill);
            }
        }
    }
}



void tensor_copy_row_data(Tensor *dest_tensor, size_t batch_id, size_t row_id, Tensor *src_tensor, size_t src_row, size_t no_of_items){
    size_t dest_index = batch_id * dest_tensor->stride[0] + row_id * dest_tensor->stride[1];
    size_t src_index =  src_row * (src_tensor->ndim == 3 ? src_tensor->stride[1]: src_tensor->stride[0]);
    // printf("batch_id %zu, dest_tensor->stride[0]: %u, row_id: %zu,  dest_tensor->stride[1]: %u\n", batch_id, dest_tensor->stride[0], row_id, dest_tensor->stride[1]);
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

    Tensor output = tensor_init(
        NULL, 
        x->ndim ==3 ? (uint32_t[]){x->shape[0], x->shape[1], 2}:  (uint32_t[]){x->shape[0], 2}, 
        x->ndim,
        x->dtype,  
        "tensor_mean_var.output"
    );
    size_t batch_size = x->ndim == 3 ? x->shape[0]: 1;
    size_t rows = x->ndim == 3 ? x->shape[1]: x->shape[0];
    size_t cols = x->ndim == 3 ? x->shape[2]: x->shape[1];
    //printf("batch_size: %zu, rows: %zu, cols: %zu, x->ndim: %zu\n", batch_size, rows, cols, x->ndim);
    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            float mean = 0;
            for(size_t k = 0; k < cols; k++){
                float elem = tensor_get_elem(x, x->ndim ==3 ? (uint32_t[]){i, j, k}:  (uint32_t[]){j, k});
                mean += elem;
            }
            mean = mean / cols;
            printf("%f\n", mean);

            float variance = 0;
            for(size_t k = 0; k < cols; k++){
                float elem = tensor_get_elem(x, x->ndim ==3 ? (uint32_t[]){i, j, k}:  (uint32_t[]){j, k});
                float squared_deviation = pow(elem - mean, 2);
                //float sqrt_squared_deviation = sqrt(squared_deviation);
                variance += squared_deviation;
                //printf("elem: %.3f, mean: %.3f, elem-mean: %.3f, squared_deviation: %.3f, sqrt_squared_deviation: %.3f\n", elem, mean, elem - mean, squared_deviation, sqrt_squared_deviation);
            }
            variance = variance / cols; 
            tensor_put_elem(&output, x->ndim ==3 ? (uint32_t[]){i, j, 0}:  (uint32_t[]){j, 0}, mean);
            tensor_put_elem(&output, x->ndim ==3 ? (uint32_t[]){i, j, 1}:  (uint32_t[]){j, 1}, variance);
        }
    }
    return output;
}


void tensor_mean_var_(Tensor *input, Tensor *output){

    if(output->size == 0){
        tensor_init_(
            output,
            NULL, 
            input->ndim ==3 ? (uint32_t[]){input->shape[0], input->shape[1], 2}:  (uint32_t[]){input->shape[0], 2}, 
            input->ndim,
            input->dtype,  
            NULL
        );
    }
    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
    assert(cols > 0);
    //printf("batch_size: %zu, rows: %zu, cols: %zu, x->ndim: %zu\n", batch_size, rows, cols, x->ndim);
    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double mean = 0;
            for(size_t k = 0; k < cols; k++){
                float elem = tensor_get_elem(input, input->ndim ==3 ? (uint32_t[]){i, j, k}:  (uint32_t[]){j, k});
                if (isnan(elem)) {
                    printf("elem is nan\n");
                    exit(1);
                }
                if (isinf(elem)) {
                    printf("elem is inf\n");
                    exit(1);
                }
                mean += elem;
            }
            mean = mean / cols;

            double variance = 0;
            for(size_t k = 0; k < cols; k++){
                float elem = tensor_get_elem(input, input->ndim ==3 ? (uint32_t[]){i, j, k}:  (uint32_t[]){j, k});
                double squared_deviation = pow(elem - mean, 2);
                //float sqrt_squared_deviation = sqrt(squared_deviation);
                variance += squared_deviation;
                //printf("elem: %.3f, mean: %.3f, elem-mean: %.3f, squared_deviation: %.3f, sqrt_squared_deviation: %.3f\n", elem, mean, elem - mean, squared_deviation, sqrt_squared_deviation);
            }
            variance = variance / cols; 
            tensor_put_elem(output, input->ndim ==3 ? (uint32_t[]){i, j, 0}:  (uint32_t[]){j, 0}, mean);
            tensor_put_elem(output, input->ndim ==3 ? (uint32_t[]){i, j, 1}:  (uint32_t[]){j, 1}, variance);
            if (isnan(mean)) {
                printf("mean is nan\n");
                exit(1);
            }
            if (isinf(mean)) {
                printf("mean is inf\n");
                exit(1);
            }

            if (isnan(variance)) {
                printf("variance is nan\n");
                exit(1);
            }
            if (isinf(variance)) {
                printf("variance is inf\n");
                exit(1);
            }
        }
    }
}

Tensor tensor_norm(Tensor *x, Tensor *mean_var_tensor, float eps){
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
            float mean = tensor_get_elem(mean_var_tensor, x->ndim == 3 ? (uint32_t[]){i, j, 0}: (uint32_t[]){j, 0});
            float var = tensor_get_elem(mean_var_tensor, x->ndim == 3 ? (uint32_t[]){i, j, 1}: (uint32_t[]){j, 1});
            //printf("mean: %.2f, var: %.2f\n", mean, var);
            for(size_t k = 0; k < cols; k++){
                float elem = tensor_get_elem(x, x->ndim == 3? (uint32_t[]){i, j, k}: (uint32_t[]){j, k});
                float norm_elem = (elem - mean) / sqrt(var + eps);
                tensor_put_elem(&output_tensor, x->ndim == 3 ? (uint32_t[]){i, j, k}: (uint32_t[]){j, k}, norm_elem);
            }
        }
    }
    return output_tensor;
}


void tensor_norm_(Tensor *input, Tensor *mean_var_tensor, float eps, Tensor *output){
    assert(input->ndim == mean_var_tensor->ndim);
    //assert(mean_var_tensor->shape[mean_var_tensor->ndim-1] == 2);
    assert(input->shape[0] == mean_var_tensor->shape[0]);
    if(input->ndim == 3) assert(input->shape[1] == mean_var_tensor->shape[1]);

    if(output->size == 0){
        tensor_init_(
            output,
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
            float mean = tensor_get_elem(mean_var_tensor, input->ndim == 3 ? (uint32_t[]){b, i, 0}: (uint32_t[]){i, 0});
            float var = tensor_get_elem(mean_var_tensor, input->ndim == 3 ? (uint32_t[]){b, i, 1}: (uint32_t[]){i, 1});
            //printf("mean: %.2f, var: %.2f\n", mean, var);
            for(size_t j = 0; j < cols; j++){
                float elem = tensor_get_elem(input, input->ndim == 3? (uint32_t[]){b, i, j}: (uint32_t[]){i, j});
                float norm_elem = (elem - mean) / sqrt(var + eps);
                tensor_put_elem(output, input->ndim == 3 ? (uint32_t[]){b, i, j}: (uint32_t[]){i, j}, norm_elem);
            }
        }
    }
}


bool tensor_isnan(Tensor *x){
    for(size_t i = 0; i < x->size; i++){
        if(x->dtype == DTYPE_FP32){
            if(isnan(((float*)x->data)[i])) return true;
            if(isinf(((float*)x->data)[i])) return true;
        }
        else if(x->dtype == DTYPE_INT32){
            if(isnan(((int*)x->data)[i])) return true;
            if(isinf(((int*)x->data)[i])) return true;
        }
    }
    return false;
}
void tensor_print(const Tensor *tensor, const char *heading){
    #define DEBUG
    #ifdef DEBUG    
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
    printf("name:           %s\n", tensor->name);

    printf("data:\n");

    size_t size = tensor->size > 10 ? 10: tensor->size;
    for(size_t i = 0; i < size; i++){
        if(tensor->dtype == DTYPE_FP32){
            float elem = ((float*)tensor->data)[i];
            if(elem == -FLT_MAX){
                printf("%s    ", "-INF");
            }
            else{
                printf("%.4f    ", elem);
            }
        }
        else if(tensor->dtype == DTYPE_INT32) printf("%d    ", ((int*)tensor->data)[i]);
    }
    // printf("\n");
    // for(size_t i = tensor->size-1; i >= tensor->size-size; i--){
    //     if(tensor->dtype == DTYPE_FP32){
    //         float elem = ((float*)tensor->data)[i];
    //         if(elem == -FLT_MAX){
    //             printf("%s    ", "-INF");
    //         }
    //         else{
    //             printf("%.2f    ", elem);
    //         }
    //     }
    //     else if(tensor->dtype == DTYPE_INT32) printf("%d    ", ((int*)tensor->data)[i]);
    // }
    printf("\n");
    // if(tensor->ndim == 4){
    //     // uint32_t shape_i = tensor->shape[0];
    //     // uint32_t shape_j = tensor->shape[1];
    //     // uint32_t shape_k = tensor->shape[2];
    //     // uint32_t shape_l = tensor->shape[3];

    //     uint32_t shape_i = 1;
    //     uint32_t shape_j = 1;
    //     uint32_t shape_k = 4;
    //     uint32_t shape_l = 4;
    //     printf("[\n");
    //     for(size_t i = 0; i < shape_i; i++){
    //         printf("    [\n");
    //         for(size_t j = 0; j < shape_j; j++){
    //             printf("        [\n");
    //             for(size_t k = 0; k < shape_k; k++){
    //                 //printf("            [\n");
    //                 for(size_t l = 0; l < shape_l; l++){
    //                     float elem = tensor_get_elem(tensor, (uint32_t[]){i, j, k, l});
    //                     if(tensor->dtype == DTYPE_FP32) printf("            %5.2f ", elem);
    //                     else if(tensor->dtype == DTYPE_INT32) printf("      %d ", (int)elem);
    //                 }
    //                 printf("\n");
    //             }
    //             printf("        ]\n");
    //         }
    //         printf("    ]\n");
    //     }
    //     printf(" ]\n");
    // }
    
    // if(tensor->ndim == 3){
    //     uint32_t shape_i = tensor->shape[0];
    //     uint32_t shape_j = tensor->shape[1];
    //     uint32_t shape_k = tensor->shape[2];
    //     printf("[\n");
    //     for(size_t i = 0; i < shape_i; i++){
    //         for(size_t j = 0; j < shape_j; j++){
    //             printf("    [ ");
    //             for(size_t k = 0; k < shape_k; k++){
    //                 float elem = tensor_get_elem(tensor, (uint32_t[]){i, j, k});
    //                 if(tensor->dtype == DTYPE_FP32){ 
    //                     if(elem == -FLT_MAX){
    //                         printf("%s    ", "-INF");
    //                     }
    //                     else{
    //                         printf("%10.2f ", elem);
    //                     }
    //                 }
    //                 else if(tensor->dtype == DTYPE_INT32) printf("%d   ", (int)elem);
    //             }
    //             printf(" ]\n");
    //         }
    //         // if(i <= shape_i - 2) printf("\n\n");
    //     }
    //     printf("]\n");
    // }
    // else if(tensor->ndim == 2){
    //     uint32_t shape_i = tensor->shape[0];
    //     uint32_t shape_j = tensor->shape[1];
    //     //printf("shape_i: %u, shape_j: %u ", shape_i, shape_j);
    //     for(size_t i = 0; i < shape_i; i++){
    //         printf("[ ");
    //         for(size_t j = 0; j < shape_j; j++){
    //             float elem = tensor_get_elem(tensor, (uint32_t[]){i, j});
    //             if(tensor->dtype == DTYPE_FP32) printf("%10.2f ", elem);
    //             else if(tensor->dtype == DTYPE_INT32) printf("%d    ", (int)elem);
    //         }
    //         printf(" ]\n");
    //     }
    // }
    // else if(tensor->ndim == 1){
    //     printf("[ ");
    //     for(size_t i = 0; i < 10; i++){
    //         float elem = tensor_get_elem(tensor, (uint32_t[]){i});
    //         if(tensor->dtype == DTYPE_FP32) printf("%10.2f ", elem);
    //         else if(tensor->dtype == DTYPE_INT32) printf("%d    ", (int)elem);
    //     }
    //     printf(" ]\n");
    // }
    // printf("\n");
    #endif
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
//                     float elem = tensor_get_elem(tensor, (uint32_t[]){b, i, j});
//                     if(tensor->dtype == DTYPE_FP32) fprintf(fptr, ",%.17g", elem);
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

//     uint32_t t_size =  t->size * (t->elem_size); 
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




