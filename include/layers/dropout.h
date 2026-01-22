#ifndef __DROPOUT_H__
#define __DROPOUT_H__

#include <stdbool.h>
#include "./tensor.h"



typedef struct DropoutLayer{
    double dropout;
    bool eval;
} DropoutLayer;


DropoutLayer dropout_layer_init(const double dropout, const bool eval);
void dropout_layer_forward(const DropoutLayer *dropout_layer, Tensor *tensor);

#endif