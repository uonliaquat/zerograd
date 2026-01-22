#include "../../include/layers/dropout.h"
#include "../../include/utils.h"
#include <assert.h>

DropoutLayer dropout_layer_init(const double dropout, const bool eval){
    DropoutLayer dropout_layer;
    dropout_layer.dropout = dropout;
    dropout_layer.eval = eval;
    return dropout_layer;
}

void dropout_layer_forward(const DropoutLayer *dropout_layer, Tensor *tensor){
    assert(tensor->ndim == 2);
    if(dropout_layer->eval == false){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            for(size_t j = 0; j < tensor->shape[1]; j++){
                double rand_elem = rand_double(0.0, 1.0);
                if(rand_elem < dropout_layer->dropout){
                    tensor_put_elem(tensor, (size_t[]){i, j}, 0.0);
                }
            }
        }
    }
}