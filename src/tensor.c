#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/tensor.h"
#include "../include/utils.h"

Tensor tensor_init(const void *data, const size_t * shape, const size_t ndim, const size_t elem_size, const bool requires_grad, const bool random_init){

    Tensor tensor = { 0 };




    tensor.ndim = ndim;
    tensor.elem_size = elem_size;
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
    tensor.stride[3] = 1;
    //memset(tensor.stride, 0, sizeof(tensor.stride));
    for(size_t i = 0; i < ndim-1; i++){
        tensor.stride[i] = tensor.shape[i+1];
    }

    // tensor.stride[3] = 1;
    // for(size_t i = 3; i > 0; i--){
    //     tensor.stride[i-1] = tensor.shape[i];
    // }

    // Initializing data
    tensor.data = (void*)calloc(tensor.size, elem_size);
    if(data != NULL)
        memcpy(tensor.data, data, elem_size * tensor.size);
    else if(random_init == true){
        for(size_t i = 0; i < tensor.size; i++){
            double rand_number = rand_double(-10.0, 10.0);
            ((double*)tensor.data)[i] = rand_number;
        }
    }
    


    // Initializing shape
    // memset(tensor.shape[0], 1, sizeof(tensor.shape[0]));
    // memset(tensor.shape[1], 1, sizeof(tensor.shape[1]));
    // memset(tensor.shape[2], 1, sizeof(tensor.shape[2]));
    // memset(tensor.shape[3], 1, sizeof(tensor.shape[3]));
    // memcpy(tensor.shape, shape, 8);

    

    return tensor;
}

void tensor_free(const Tensor *tensor){
    free(tensor->data);
}



void tensor_print(const Tensor *tensor){
    printf("\n\n");
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
    printf("data:\n");


    
    for(size_t i = 0; i < tensor->shape[0]; i++){    
        if(tensor->shape[1] != 1)
            printf("[ ");
        for(size_t j = 0; j < tensor->shape[1]; j++){
            int val_index = i * tensor->stride[0] + j * tensor->stride[1];
            printf("%.2f ", ((double*)tensor->data)[val_index]);
        }
        printf(" ]\n");
    }

    printf("requires_grad:  %s\n", tensor->requires_grad ? "true" : "false");

    printf("\n\n");
}


