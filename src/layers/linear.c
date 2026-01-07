#include "../../include/layers/linear.h"

#include <stdlib.h>
LinearLayer linear_layer_init(const size_t inputs, const size_t outputs, const bool bias, const DataType dtype){
    LinearLayer linear_layer;
    linear_layer.weights  = tensor_init(NULL, (size_t[]){outputs, inputs}, 2, dtype, false, true);
    if(bias == true)
        linear_layer.bias     = tensor_init(NULL, (size_t[]){outputs, 1}, 2, dtype, false, true);
    linear_layer.dtype = dtype;
    return linear_layer;
}