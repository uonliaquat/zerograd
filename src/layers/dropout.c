#include "../../include/layers/dropout.h"
#include "../../include/utils.h"
#include <assert.h>

DropoutLayer dropout_layer_init(const double dropout, const bool eval){
    DropoutLayer dropout_layer;
    dropout_layer.dropout = dropout;
    dropout_layer.eval = eval;
    return dropout_layer;
}

static inline double dropout(double x, double dropout){
    double u = rand_uniform(0.0, 1.0);
    if(u < dropout) return 0.0;
    else{
        return x / (1.0 - dropout);
    }
}
void dropout_layer_forward(const DropoutLayer *dropout_layer, Tensor *tensor){
    assert(tensor->ndim == 3);
    if(dropout_layer->eval == false){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            for(size_t j = 0; j < tensor->shape[1]; j++){
                for(size_t k = 0; k < tensor->shape[2]; k++){
                    double x = tensor_get_elem(tensor, (size_t[]){i, j, k});
                    tensor_put_elem(tensor, (size_t[]){i, j, k}, dropout(x, dropout_layer->dropout));
                }   
            }
        }
    }
}