#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"


int main(){
    printf("Running Main...\n\n");
    float data[] = {1, 2, 3, 4, 5};
    size_t shape[] = {5, 1, 1, 1}; 
    Tensor tensor = tensor_init(data, shape, sizeof(float), false);
    tensor_print(&tensor);
    
    // Taking dot product
    float A[2*3] = {
        1, 2, 3,
        4, 5, 6
    };
    float B[3*2] = {
        7,  8,
        9, 10,
        11,12
    };
    float C[2*2] = {0};

    dot_product(A, B, C, /*m=*/2, /*n=*/2, /*k=*/3);

    printf("[ %g %g ]\n", C[0], C[1]);   // row 0
    printf("[ %g %g ]\n", C[2], C[3]);   // row 1

    return 0;
}
