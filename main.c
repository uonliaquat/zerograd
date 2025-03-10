//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>

#include "./include/tensor.h"



int main(){
    srand(42);

    const size_t tensor_size = 1024;
    Tensor tensor_a = create_tensor(tensor_size, tensor_size, FLOAT32);
    Tensor tensor_b = create_tensor(tensor_size, tensor_size, FLOAT32);
    Tensor tensor_c = create_tensor(tensor_size, tensor_size, FLOAT32);
    init_tensor_rand(&tensor_a);
    init_tensor_rand(&tensor_b);
    init_tensor_rand(&tensor_c);
    

    gemm(&tensor_a, &tensor_b, &tensor_c);


    //print_tensor(&tensor_a);
    //print_tensor(&tensor_b);
    //print_tensor(&tensor_c);

    // save tensors
    save_tensor(&tensor_a, "./a_mat.csv");
    save_tensor(&tensor_b, "./b_mat.csv");
    save_tensor(&tensor_c, "./c_mat.csv");


    return 0;
}






