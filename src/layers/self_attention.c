
#include <stddef.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "../../include/layers/self_attention.h"
#include "../../include/layers/linear.h"


SelfAttentionLayer self_attention_layer_init(const size_t inputs, const size_t outputs, const bool bias, const bool requires_grad, const DataType dtype){
    SelfAttentionLayer self_attention_layer;
    self_attention_layer.W_query = linear_layer_init(inputs, outputs, bias, requires_grad, dtype);
    self_attention_layer.W_key = linear_layer_init(inputs, outputs, bias, requires_grad, dtype);
    self_attention_layer.W_value = linear_layer_init(inputs, outputs, bias, requires_grad, dtype);
    return self_attention_layer;
}

Tensor self_attention_layer_forward(const SelfAttentionLayer *self_attention_layer, const Tensor *x){
    Tensor W_query_output = linear_layer_forward(&self_attention_layer->W_query, x);
    Tensor W_key_output = linear_layer_forward(&self_attention_layer->W_key, x);
    Tensor W_value_output = linear_layer_forward(&self_attention_layer->W_value, x);

    // attenion_scroes = Q.K^t
    Tensor W_key_output_transposed = tensor_transpose(&W_key_output);
    Tensor attention_scores = tensor_dot_product_matrix(&W_query_output, &W_key_output_transposed);
    Tensor attention_scores_scaled = tensor_mul(&attention_scores, 1/sqrt(W_key_output.shape[1]));
    Tensor attention_scores_normalized = tensor_softmax(&attention_scores_scaled, 1);
    Tensor output = tensor_dot_product_matrix(&attention_scores_normalized, &W_value_output);
    return output;

}

void self_attention_layer_print(const SelfAttentionLayer *self_attention_layer, const char *heading){
    printf("\033[36m==============================LINEAR LAYER %s (WEIGHTS)==============================\033[0m\n", heading);
    linear_layer_print(&self_attention_layer->W_query, "W_Query");
    linear_layer_print(&self_attention_layer->W_key, "W_key");
    linear_layer_print(&self_attention_layer->W_value, "W_value");
}


void self_attention_layer_write(const SelfAttentionLayer *self_attention_layer, const char *filename){
    FILE *fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Couldn't open file %s\n", filename);
        exit(1);
    }
    fprintf(fptr, "%s\n", "W_Query");
    linear_layer_write_fp(&self_attention_layer->W_query, fptr);
    fprintf(fptr, "%s\n", "W_Key");
    linear_layer_write_fp(&self_attention_layer->W_key, fptr);
    fprintf(fptr, "%s\n", "W_value");
    linear_layer_write_fp(&self_attention_layer->W_value, fptr);
}


// Tensor self_attention_simplified(Tensor *input_embeddings){

//     Tensor input_embeddings_transposed = tensor_transpose(input_embeddings);
//     tensor_print(&input_embeddings_transposed);
//     tensor_write(&input_embeddings_transposed, "./output/input_embeddings_transposed.csv");

//     Tensor attention_scores = tensor_dot_product_matrix(input_embeddings, &input_embeddings_transposed);
//     tensor_print(&attention_scores);

//     Tensor attention_weights = tensor_softmax(&attention_scores, 1);
//     tensor_print(&attention_weights);

//     Tensor context_vectors = tensor_dot_product_matrix(&attention_weights, input_embeddings);
//     tensor_print(&context_vectors);

//     return input_embeddings_transposed;
// }
