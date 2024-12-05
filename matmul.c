#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define A_ROWS 100
#define A_COLS 100
#define B_ROWS A_COLS
#define B_COLS 100
#define C_ROWS A_ROWS
#define C_COLS B_COLS



void matmul_1(float *, float *, float *);
void init_mat(float *, unsigned int, unsigned int);
void save_mat(float *, unsigned int, unsigned int, char *);
void print_mat(float *, unsigned int, unsigned int);
float generate_random_float(float, float);

int main(){
    srand(42);

    static float a[A_ROWS*A_COLS] = {0};
    static float b[B_ROWS*B_COLS] = {0};
    static float c[C_ROWS*C_COLS] = {0};
    
    // initialze matrices
    init_mat(a, A_ROWS, A_COLS);
    init_mat(b, B_ROWS, B_COLS);

    // save matrices
    save_mat(a, A_ROWS, A_COLS, "./a_mat.csv");
    save_mat(b, B_ROWS, B_COLS, "./b_mat.csv");

    matmul_1(a, b, c);
    save_mat(c, C_ROWS, C_COLS, "./c_mat.csv");
    //print_mat(c, C_ROWS, C_COLS);

    return 0;
}

float generate_rand_val(float lower, float upper){
    return lower + ((float)rand() / RAND_MAX) * (upper - lower);
}

float get_element(float *mat, unsigned int rows, unsigned int cols, unsigned int row, unsigned int col){
    return mat[(row * cols) + col];
}

void put_element(float *mat, unsigned int rows, unsigned int cols, unsigned int row, unsigned int col, float val){
    mat[(row * cols) + col] = val;
}

void init_mat(float *mat, unsigned int rows, unsigned int cols){
    for(unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < cols; j++){
            float rand_val = generate_rand_val(-10, 10);
            put_element(mat, rows, cols, i, j, rand_val);
        }
    }
}

void print_mat(float *mat, unsigned int rows, unsigned int cols){
    for(unsigned int i = 0; i < rows; i++){
        printf("\n");
        for(unsigned int j = 0; j < cols; j++){
            printf("|   %f  ", get_element(mat, rows, cols, i, j));
        }
        printf("|\n");
    }
}

void save_mat(float *mat, unsigned int rows, unsigned int cols, char *file_path){
    FILE *file = fopen(file_path, "w");
    if (file == NULL){
        printf("Error opening file %s\n", file_path);
        return;
    }
    for(unsigned int i = 0; i < rows; i++){
        for(unsigned int j = 0; j < cols; j++){
            fprintf(file, "%.6f", get_element(mat, rows, cols, i, j));
            if (j != cols - 1)
                fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
}

void matmul_1(float *a, float *b, float *c){ 
    clock_t start = clock();
    for(unsigned int i = 0; i < C_ROWS; i++){
        for(unsigned int j = 0; j < C_COLS; j++){
            float val_c = 0;
            for(unsigned int k = 0; k < A_COLS; k++){
                float val_a = get_element(a, A_ROWS, A_COLS, i, k);
                float val_b = get_element(b, B_ROWS, B_COLS, k, j);
                val_c += val_a * val_b;
            }
            put_element(c, C_ROWS, C_COLS, i, j, val_c);
        }
    }
    clock_t end = clock();
    double time_spent = (double)(end - start)/CLOCKS_PER_SEC; 
    printf("Time taken: %.6f seconds\n", time_spent);
}
