#include "../../include/layers/linear.h"

#include <stdlib.h>
LinearLayer linear_layer_init(const size_t inputs, const size_t outputs, const bool bias, const bool requires_grad, const DataType dtype){
    LinearLayer linear_layer;
    linear_layer.weights  = tensor_init(NULL, (size_t[]){inputs, outputs}, 2, dtype, requires_grad, true);
    linear_layer.has_bias = bias;
    if(bias == true)
        linear_layer.bias = tensor_init(NULL, (size_t[]){1, outputs}, 2, dtype, requires_grad, true);
    linear_layer.dtype = dtype;
    return linear_layer;
}

Tensor linear_layer_forward(const LinearLayer *linear_layer, const Tensor *x){
    // Y = X.W
   //Tensor transposed_weight_matrix = tensor_transpose(&linear_layer->weights);
    Tensor a = tensor_dot_product(x, &linear_layer->weights);
    // if(linear_layer->has_bias == true){
    //     Tensor z = tensor_add(&a, &linear_layer->bias);
    //     return z;
    // }
    return a;
}

void linear_layer_print(const LinearLayer *layer, const char *heading){
    printf("\033[33m============================== LINEAR LAYER %s ==============================\033[0m", heading);
    tensor_print(&layer->weights, "Weights");
}

void linear_layer_write_fp(const LinearLayer *layer, FILE *fptr){
    tensor_write_fp(&layer->weights, fptr);
}

//6x4 . 4x2 -> 6x2