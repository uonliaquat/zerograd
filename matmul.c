#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define GET_ELEMENT(tensor, row, col) \
    (tensor->dtype == FLOAT32) ? \
        (((float*)tensor->data)[row * tensor->cols + col]) : \
        (((double*)tensor->data)[row* tensor->cols + col])

#define PUT_ELEMENT(tensor, row, col, val) \
    (tensor->dtype == FLOAT32) ? \
        (((float*)tensor->data)[row * tensor->cols + col] = val) : \
        (((double*)tensor->data)[row * tensor->cols + col] = val)




typedef enum DataType{
    FLOAT32,
    FLOAT64
} DataType;

typedef struct Tensor{
    void *data;
    DataType dtype;
    size_t rows;
    size_t cols;
} Tensor;



//void matmul(float *, float *, float *);


// Tensor Functions
Tensor create_tensor(const size_t, const size_t, DataType);
void init_tensor_rand(Tensor*);
void gemm(Tensor *, Tensor *, Tensor *);
void print_tensor(Tensor *);
void save_tensor(Tensor *, char *);

// Dot Product Functions
void (*dot)(Tensor *, Tensor *, Tensor *);


// Utils
static inline float generate_rand_val(float, float);
//float cal_flops(unsigned int, double);



int main(){
    srand(42);

    const size_t tensor_size = 1024;
    Tensor tensor_a = create_tensor(tensor_size, tensor_size, FLOAT32);
    Tensor tensor_b = create_tensor(tensor_size, tensor_size, FLOAT32);
    Tensor tensor_c = create_tensor(tensor_size, tensor_size, FLOAT32);
    init_tensor_rand(&tensor_a);
    init_tensor_rand(&tensor_b);
    init_tensor_rand(&tensor_c);

    dot = gemm;
    dot(&tensor_a, &tensor_b, &tensor_c);


    //print_tensor(&tensor_a);
    //print_tensor(&tensor_b);
    //print_tensor(&tensor_c);

    // save tensors
    save_tensor(&tensor_a, "./a_mat.csv");
    save_tensor(&tensor_b, "./b_mat.csv");
    save_tensor(&tensor_c, "./c_mat.csv");


    return 0;
}


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


void gemm(Tensor *a, Tensor *b, Tensor *c){
    clock_t start = clock();
    for(size_t row_c = 0; row_c < c->rows; row_c++){
        for(size_t col_c = 0; col_c < c->cols; col_c++){
            double val_c = 0;
            for(size_t col_a = 0; col_a < a->cols; col_a++){
                float val_a = GET_ELEMENT(a, row_c, col_a);
                float val_b = GET_ELEMENT(b, col_a, col_c);
                val_c += (val_a * val_b);
            }
            PUT_ELEMENT(c, row_c, col_c, val_c);
        }
    }
    clock_t end = clock();
    double time_seconds = (double)(end - start)/CLOCKS_PER_SEC;
    double flop = 2*(c->rows * c->cols * a->cols);
    double gflops = (flop / time_seconds) * 1e-9;
    printf("Time(s): %.4f \n", time_seconds);
    printf("GFLOPS:  %.4f \n", gflops);
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

static inline float generate_rand_val(float lower, float upper){
    return lower + ((float)rand() / RAND_MAX) * (upper - lower);
}
/*
static inline float cal_flops(unsigned int operations, double time_s){
    return operations / time_s;
}








*/

