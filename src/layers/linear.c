#include "../../include/layers/linear.h"

#include <stdlib.h>
LinearLayer linear_layer_init(const size_t inputs, const size_t outputs, const bool bias, const bool requires_grad, const DataType dtype){
    LinearLayer linear_layer;
    linear_layer.weights  = tensor_init(NULL, (uint32_t[]){inputs, outputs}, 2, dtype, NULL);
    linear_layer.has_bias = bias;
    if(bias == true)
        linear_layer.bias = tensor_init(NULL, (uint32_t[]){1, outputs}, 2, dtype, NULL);
    linear_layer.dtype = dtype;
    linear_layer.output = (Tensor){0};
    return linear_layer;
}

void linear_layer_free(const LinearLayer *linear_layer){
    tensor_free(&linear_layer->weights);
    tensor_free(&linear_layer->output);
}

void linear_layer_forward(LinearLayer *linear_layer, const Tensor *x){
    // Y = X.W
   //Tensor transposed_weight_matrix = tensor_transpose(&linear_layer->weights);
   //if(linear_layer->output.size == 0) linear_layer->output = tensor_dot_product(x, &linear_layer->weights);
   tensor_dot_product_(x, &linear_layer->weights, &linear_layer->output);
    // if(linear_layer->has_bias == true){
    //     Tensor z = tensor_add(&a, &linear_layer->bias);
    //     return z;
    // }
}

void linear_layer_print(const LinearLayer *layer, const char *heading){
    printf("\033[33m============================== LINEAR LAYER %s ==============================\033[0m", heading);
    tensor_print(&layer->weights,   "Weights");
    tensor_print(&layer->output,    "Output");
}

// void linear_layer_write_fp(const LinearLayer *layer, FILE *fptr){
//     fprintf(fptr, "Weights\n");
//     tensor_write_fp(&layer->weights, fptr);
//     fprintf(fptr, "Output\n");
//     tensor_write_fp(&layer->output, fptr);
// }

// void linear_layer_write(const LinearLayer *layer, const char *filename){
//     FILE *fptr = fopen(filename, "w");
//     fclose(fptr);
//     fptr = fopen(filename, "a");
//     if(fptr == NULL){
//         printf("Error Opening file %s\n", filename);
//     }
//     linear_layer_write_fp(layer, fptr);
//     fclose(fptr);
// }

//6x4 . 4x2 -> 6x2