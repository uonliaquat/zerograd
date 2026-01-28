#include <stdlib.h>
#include <string.h>
#include "../../include/layers/layer_norm.h"


LayerNorm layer_norm_init(size_t embed_dim){
    LayerNorm layer_norm;
    layer_norm.eps = 1e-5;
    double *ones = (double*)calloc(embed_dim, sizeof(double));
    for(size_t i = 0; i < embed_dim; i++) ones[i] = 1.0;
    layer_norm.scale = tensor_init(ones, (size_t[]){embed_dim, 1}, 2, DTYPE_DOUBLE, false, false);
    free(ones);
    
    layer_norm.shift = tensor_init(NULL, (size_t[]){embed_dim, 1}, 2, DTYPE_DOUBLE, false, false);

    return layer_norm;
}

Tensor layer_norm_forward(LayerNorm *layer_norm, Tensor *x){
    Tensor mean_var = tensor_mean_var(x);
    tensor_print(&mean_var, "mean_var");
    Tensor x_norm = tensor_norm(x, &mean_var,  layer_norm->eps);
    tensor_print(&x_norm, "x_norm");
    tensor_print(&layer_norm->scale, "layer_norm->scale,");
    Tensor x_norm_scaled = tensor_vector_scale(&x_norm, &layer_norm->scale);
    tensor_print(&x_norm_scaled, "x_norm_scaled");
    Tensor x_norm_scaled_shifted = tensor_vector_add(&x_norm_scaled, &layer_norm->shift);
    tensor_print(&x_norm_scaled_shifted, "x_norm_scaled_shifted");
    return mean_var;
}