#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <stdio.h>
#include "../tensor.h"

typedef struct LinearLayer{
    Tensor weights;
    Tensor bias;

}LinearLayer;

LinearLayer linear_layer_init(const size_t inputs, const size_t outputs, const bool bias);
void linear_layer_free();

#endif