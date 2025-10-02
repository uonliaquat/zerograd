#include "../../include/layers/linear.h"

#include <stdlib.h>
LinearLayer linear_layer_init(const size_t inputs, const size_t outputs, const bool bias){
    LinearLayer linear_layer;
    linear_layer.weights  = tensor_init(NULL, (size_t){outputs, inputs}, 2, sizeof(double), false, true);
    if(bias == true)
        linear_layer.bias     = tensor_init(NULL, (size_t){outputs, 1}, 2, sizeof(double), false, true);
    return linear_layer;
}