#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"


Tensor tensor_init(const void *data, const size_t * shape, const size_t elem_size, bool requires_grad){

    Tensor tensor = { 0 };
    tensor.elem_size = elem_size;                       // Element Size
    memcpy(tensor.shape, shape, sizeof(tensor.shape));  // Initializing shape
    
    // Initializing total no of elements and total dimensions 
    tensor.size = 1;
    for(size_t i = 0; i < 4; i++){
        if(tensor.shape[i] > 1)
            tensor.ndim += 1;
        tensor.size = tensor.size * tensor.shape[i];
    }


    // Initializing stride
    tensor.stride[3] = 1;
    for(size_t i = 3; i > 0; i--){
        tensor.stride[i-1] = tensor.shape[i];
    }

    
    // Initializing data
    tensor.data = (void*)calloc(tensor.size, elem_size);
    memcpy(tensor.data, data, elem_size * tensor.size);

    tensor.requires_grad = requires_grad;
    

    return tensor;
}

void tensor_free(const Tensor *tensor){
    free(tensor->data);
}



void tensor_print(const Tensor *tensor){
    printf("size:           %zu\n", tensor->size);
    printf("ndim:           %zu\n", tensor->ndim);
    
    printf("shape:          ( ");
    for(size_t i = 0; i < 4; i++){
        if(tensor->shape[i] != 0)
            printf("%zu, ", tensor->shape[i]);
    }
    printf(" )\n");
    printf("stride:         ( ");
    for(size_t i = 0; i < 4; i++){
        if(tensor->stride[i] != 0)
            printf("%zu, ", tensor->stride[i]);
    }
    printf(" )\n");
    printf("elem_size:      %zu\n", tensor->elem_size);
    printf("data:           [ ");
    for(size_t i = 0; i < tensor->size; i++){
        printf("%f ", ((float*)tensor->data)[i]);
    }
    printf(" ]\n");

    printf("requires_grad:  %s\n", tensor->requires_grad ? "true" : "false");

    printf("\n\n");
}


