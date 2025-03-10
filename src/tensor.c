#include <stdio.h>

#include "../include/tensor.h"

Tensor create_tensor(size_t rows, size_t cols, DataType datatype){
    size_t element_size = (datatype == FLOAT32) ? 4 : 8;
    void *data = (void*)calloc(rows * cols, element_size);
    if (data == NULL){
        printf("Memory Allocation failed for Tensor of size %ldx%ld\n", rows, cols);
    }
    Tensor tensor = {data, datatype, rows, cols};
    return tensor;

}

void init_tensor_rand(Tensor *tensor){
    for (size_t row = 0; row < tensor->rows; row++){
        for(size_t col = 0; col < tensor->cols; col++){
            PUT_ELEMENT(tensor, row, col, generate_rand_val(-10, 10));
        }
    }
}


void print_tensor(Tensor *tensor){
    for(size_t row = 0; row < tensor->rows; row++){
        printf("\n");
        for(size_t col = 0; col < tensor->cols; col++){
            printf("| %.2f ", GET_ELEMENT(tensor, row, col));
        }
        printf("|\n");
    }
    printf("__________________\n");
}


void save_tensor(Tensor *tensor, char *file_path){
    FILE *file = fopen(file_path, "w");
    if (file == NULL){
        printf("Error opening file %s\n", file_path);
        return;
    }
    for(size_t row = 0; row < tensor->rows; row++){
        for(size_t col = 0; col < tensor->cols; col++){
            fprintf(file, "%.6f", GET_ELEMENT(tensor, row, col));
            if (col != tensor->cols - 1)
                fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
}
