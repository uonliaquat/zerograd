#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include <stdio.h>
#include "../tensor.h"

typedef struct EmbeddingLayer{
    Tensor weights;
    Tensor bias;
    DataType dtype;
} EmbeddingLayer;


EmbeddingLayer embedding_layer_init(const size_t inputs, const size_t outputs, const bool bias, const DataType dtype);
void embedding_layer_free(EmbeddingLayer *embedding_layer);
void embedding_layer_write(EmbeddingLayer *embedding_layer, const char *filename);
#endif