#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>


#include "../include/tensor.h"
#include "../include/blas.h"
#include "../include/layers/linear.h"

int main(){
    // printf("Testing Tensors...\n\n");
    
    // // Seed the random number generator
    // srand((unsigned int)time(NULL));

    // Tensor tensor_a = tensor_init((double[]){1,2,3,4,5,6}, (size_t[]){2,3}, 2, sizeof(double), false, false);
    // printf("tensor_a");
    // tensor_print(&tensor_a);
    
    // Tensor tensor_b = tensor_init((double[]){7,8,9,10,11,12}, (size_t[]){3,2}, 2, sizeof(double), false, false);
    // printf("tensor_b");
    // tensor_print(&tensor_b);

    
    // Tensor tensor_c = dot_product_tensor(&tensor_a, &tensor_b);
    // tensor_print(&tensor_c);



    printf("Testing layers...");
    LinearLayer layer1 = LinearLayer(3, 5, false);

    
    // Taking dot product
    // double A[2*3] = {
    //     1, 2, 3,
    //     4, 5, 6
    // };
    // double B[3*2] = {
    //     7,  8,
    //     9, 10,
    //     11,12
    // };
    // double C[2*2] = {0};

    // dot_product(A, B, C, /*m=*/2, /*n=*/2, /*k=*/3);

    // printf("[ %g %g ]\n", C[0], C[1]);   // row 0
    // printf("[ %g %g ]\n", C[2], C[3]);   // row 1

    //     int m = 2, k = 3, n = 2;
    // double alpha = 1.0, beta = 0.0;

    // // dynamically allocate
    // void *A = malloc(m * k * sizeof(double));
    // void *B = malloc(k * n * sizeof(double));
    // double *C = malloc(m * n * sizeof(double));

    // // initialize A
    // double tmpA[6] = {1, 2, 3,
    //                   4, 5, 6};
    // for (int i = 0; i < 6; i++) ((double*)A)[i] = tmpA[i];

    // // initialize B
    // double tmpB[6] = {7, 8,
    //                   9, 10,
    //                   11, 12};
    // for (int i = 0; i < 6; i++) ((double*)B)[i] = tmpB[i];

    // // initialize C
    // for (int i = 0; i < m*n; i++) C[i] = 0.0;

    // // call BLAS
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //             m, n, k,
    //             alpha, A, k,
    //             B, n,
    //             beta, C, n);

    // // print result
    // printf("[ %f %f ]\n", C[0], C[1]);
    // printf("[ %f %f ]\n", C[2], C[3]);

    // free(A);
    // free(B);
    // free(C);


    return 0;
}
