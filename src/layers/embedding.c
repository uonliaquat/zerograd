#include "../../include/layers/embedding.h"

#include <stdlib.h>

EmbeddingLayer embedding_layer_init(const size_t inputs, const size_t outputs, const bool bias, const DataType dtype){
    EmbeddingLayer embedding_layer;
    embedding_layer.weights = tensor_init(NULL, (size_t[]){outputs, inputs}, 2, dtype, false, true);
    embedding_layer.dtype = dtype;
    return embedding_layer;
}

void embedding_layer_free(EmbeddingLayer *embedding_layer){
    tensor_free(&embedding_layer->weights);
}

void embedding_layer_write(EmbeddingLayer *embedding_layer, const char *filename){
    FILE *fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    tensor_write(&embedding_layer->weights, fptr);
    fclose(fptr);
}

