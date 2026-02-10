#ifndef __MULTI_LAYER_PERCEPTRON_H__
#define __MULTI_LAYER_PERCEPTRON_H__

#include "./tensor.h"
#include "./layers/linear.h"
typedef struct MLPWorkspace{
    Tensor gelu;
    Tensor output;
} MLPWorkspace;


typedef struct MultiLayerPerceptronParams{
    LinearLayerParams c_fc;
    LinearLayerParams c_proj;
} MultiLayerPerceptronParams;

typedef struct MultiLayerPerceptron{
    char name[128];
    LinearLayer c_fc;
    LinearLayer c_proj;
    MLPWorkspace workspace;
} MultiLayerPerceptron;


MultiLayerPerceptron multi_layer_perceptron_init(MultiLayerPerceptronParams *params, char *name);
void multi_layer_perceptron_free(MultiLayerPerceptron *mlp);
void multi_layer_perceptron_forward(MultiLayerPerceptron *mlp, Tensor *x);
void multi_layer_perceptron_write(MultiLayerPerceptron *mlp, Tensor **tensors, size_t *tensors_len);

#endif