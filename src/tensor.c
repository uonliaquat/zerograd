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
    tensor.data = (void*)calloc(tensor.size,  tensor.elem_size);
    if(data != NULL)
        memcpy(tensor.data, data, tensor.elem_size * tensor.size);
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

double tensor_get_elem(const Tensor *tensor, const size_t row, const size_t col){
    size_t rows = tensor->shape[0];
    size_t cols = tensor->shape[1];
    assert(row < rows && col < cols);
    size_t index = row * tensor->stride[0] + col * tensor->stride[1];
    return ((double*)tensor->data)[index];
}

void tensor_put_elem(Tensor *tensor, size_t row, size_t col, double elem){
    size_t rows = tensor->shape[0];
    size_t cols = tensor->shape[1];
    assert(row < rows && col < cols);
    size_t index = row * tensor->stride[0] + col * tensor->stride[1];
    ((double*)tensor->data)[index] = elem;
}

Tensor tensor_transpose(const Tensor *tensor){
    Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){tensor->shape[1],  
        tensor->shape[0]}, 
        2,
        tensor->dtype,
        tensor->requires_grad,
        true
    );

    for(size_t i = 0; i < tensor->shape[0]; i++){
        for(size_t j = 0; j < tensor->shape[1]; j++){
            double elem = tensor_get_elem(tensor, i, j);
            tensor_put_elem(&output_tensor, j, i, elem);
        }
    }
    return output_tensor;
}


Tensor tensor_softmax(Tensor *tensor, size_t dim){
    Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){tensor->shape[1],  
        tensor->shape[0]}, 
        2,
        tensor->dtype,
        tensor->requires_grad,
        true
    );
    if(dim == 1){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            double exp_sum = 0;
            for(size_t j = 0; j < tensor->shape[1]; j++){
                double elem = tensor_get_elem(tensor, i, j);
                exp_sum = exp_sum + expf(elem);
            }
            for(size_t j = 0; j < tensor->shape[1]; j++){
                double elem = tensor_get_elem(tensor, i, j);

                double new_elem = expf(elem) / exp_sum;
                tensor_put_elem(&output_tensor, i, j, new_elem);
            }
        }
    }
    return output_tensor;
}

Tensor tensor_mul(Tensor *tensor, double elem){
    Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){tensor->shape[0],  
        tensor->shape[1]}, 
        2,
        tensor->dtype,
        tensor->requires_grad,
        true
    );
    for(size_t i = 0; i < tensor->shape[0]; i++){
        for(size_t j = 0; j < tensor->shape[1]; j++){
            double old_elem = tensor_get_elem(tensor, i, j);
            double new_elem = old_elem * elem;
            tensor_put_elem(&output_tensor, i, j, new_elem);
        }
    }
    return output_tensor;
}


Tensor tensor_dot_product_matrix(const Tensor *tensor1, const Tensor *tensor2){
    assert(tensor1->shape[1] == tensor2->shape[0]);
    Tensor output_tensor =  tensor_init(
        NULL, 
        (size_t[]){tensor1->shape[0],  
        tensor2->shape[1]}, 
        2,
        tensor1->dtype,
        tensor1->requires_grad,
        true
    );
    size_t t1_rows = tensor1->shape[0];
    size_t t1_cols = tensor1->shape[1];
    size_t t2_rows = tensor2->shape[0];
    size_t t2_cols = tensor2->shape[1];
    size_t out_rows = output_tensor.shape[0];
    size_t out_cols = output_tensor.shape[1];
    for(size_t i = 0; i < out_rows; i++){
        for(size_t j = 0; j < out_cols; j++){
            double result = 0;
            for(size_t k = 0; k < t1_cols; k++){
                double elem1 = tensor_get_elem(tensor1, i, k);
                double elem2 = tensor_get_elem(tensor2, k, j);
                result += elem1 * elem2;
            }
            tensor_put_elem(&output_tensor, i, j, result);
        }
    }
    return output_tensor;

}

Tensor tensor_add(Tensor *tensor1, Tensor *tensor2){
    Tensor new_tensor = tensor_init(
        tensor1->data, 
        tensor1->shape, 
        tensor1->ndim, 
        tensor1->dtype, 
        tensor1->requires_grad, 
        false
    );
    for(size_t i = 0; i < new_tensor.size; i++){
        ((double*)new_tensor.data)[i] = ((double*)new_tensor.data)[i] + ((double*)tensor2->data)[i];
    }
    return new_tensor;
}


void tensor_copy_row_data(Tensor *dest_tensor, size_t dest_row, Tensor *src_tensor, size_t src_row, size_t no_of_items){
    void *dest_data = &((double*)dest_tensor->data)[dest_row * dest_tensor->shape[1]];
    void *src_data =  &((double*)src_tensor->data)[src_row * src_tensor->shape[1]];
    memcpy(dest_data, src_data, no_of_items * src_tensor->elem_size);
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
    printf("data:\n");

    if(tensor->ndim == 1){
        printf("[ ");
        for(size_t i = 0; i < tensor->shape[0]; i++){    
            if(tensor->dtype == DTYPE_DOUBLE)
                printf("%5.2f ", ((double*)tensor->data)[i]);
            else
                printf("%5d ", ((int*)tensor->data)[i]);
        }
        printf(" ]\n");
    }
    else if(tensor->ndim == 2){
        for(size_t i = 0; i < tensor->shape[0]; i++){    
            if(tensor->shape[1] != 1)
                printf("[ ");
            for(size_t j = 0; j < tensor->shape[1]; j++){
                int val_index = i * tensor->stride[0] + j * tensor->stride[1];
                if(tensor->dtype == DTYPE_DOUBLE)
                    printf("%5.2f ", ((double*)tensor->data)[val_index]);
                else
                    printf("%5d ", ((int*)tensor->data)[val_index]);
            }
            printf(" ]\n");
        }
    }

    printf("requires_grad:  %s\n", tensor->requires_grad ? "true" : "false");

    printf("\n");
}


void tensor_write_fp(const Tensor *tensor, FILE *fptr){
    if(tensor->ndim == 1){
        for(size_t i = 0; i < tensor->shape[0]; i++){    
            if(tensor->dtype == DTYPE_DOUBLE)
                fprintf(fptr, "%5.2f,", ((double*)tensor->data)[i]);
            else
                fprintf(fptr, "%5d,", ((int*)tensor->data)[i]);
        }
        fprintf(fptr, "\n");
    }
    else if(tensor->ndim == 2){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            for(size_t j = 0; j < tensor->shape[1]; j++){
                int val_index = i * tensor->stride[0] + j * tensor->stride[1];
                if(tensor->dtype == DTYPE_DOUBLE)
                    fprintf(fptr, "%5.2f,", ((double*)tensor->data)[val_index]);
                else
                    fprintf(fptr, "%5d,", ((int*)tensor->data)[val_index]);
            }
            fprintf(fptr, "\n");
        }
    }
    fprintf(fptr, "\n");
}

void tensor_write(const Tensor *tensor, char *filename){
    FILE *fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    if(tensor->ndim == 1){
        for(size_t i = 0; i < tensor->shape[0]; i++){    
            if(tensor->dtype == DTYPE_DOUBLE)
                fprintf(fptr, "%5.2f,", ((double*)tensor->data)[i]);
            else
                fprintf(fptr, "%5d,", ((int*)tensor->data)[i]);
        }
        fprintf(fptr, "\n");
    }
    else if(tensor->ndim == 2){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            for(size_t j = 0; j < tensor->shape[1]; j++){
                int val_index = i * tensor->stride[0] + j * tensor->stride[1];
                if(tensor->dtype == DTYPE_DOUBLE)
                    fprintf(fptr, "%5.2f,", ((double*)tensor->data)[val_index]);
                else
                    fprintf(fptr, "%5d,", ((int*)tensor->data)[val_index]);
            }
            fprintf(fptr, "\n");
        }
    }
}


size_t tensor_dtype_size(const DataType dtype){
    switch (dtype){
        case DTYPE_DOUBLE: return sizeof(double);
        case DTYPE_INT: return sizeof(int);
    }
}

