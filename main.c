#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"


int main(){
    printf("Running Main...\n\n");
    float data[] = {1, 2, 3, 4, 5};
    size_t shape[] = {5, 1, 1, 1}; 
    Tensor tensor = tensor_init(data, shape, sizeof(float), false);
    tensor_print(&tensor);
    return 0;
}
