#ifndef __LINEAR_LAYER_H__
#define __LINEAR_LAYER_H__

#include <stdio.h>
#include <stdbool.h>
#include "../tensor.h"

typedef struct LinearLayer{
    Tensor weights;
    Tensor bias;
    bool has_bias;
    DataType dtype;
}LinearLayer;

LinearLayer linear_layer_init(const size_t inputs, const size_t outputs, const bool bias, const bool requires_grad, const DataType dtype);
void linear_layer_free();
Tensor linear_layer_forward(const LinearLayer *linear_layer, const Tensor *x);
void linear_layer_print(const LinearLayer *layer, const char *heading);
void linear_layer_write_fp(const LinearLayer *layer, FILE *fptr);

#endif