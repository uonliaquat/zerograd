#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include <stdio.h>
#include "../tensor.h"

#define MAX_SEQUENCE_LENGTH 100

typedef struct EmbeddingLayer{
    Tensor weights;
    Tensor output;
    DataType dtype;
    size_t vocab_size;
    size_t embed_len;
    size_t seq_len;
} EmbeddingLayer;


EmbeddingLayer  embedding_layer_init(size_t vocab_size, size_t embed_len, size_t seq_len, DataType dtype);
void            embedding_layer_free(EmbeddingLayer *embedding_layer);
void            embedding_layer_forward(EmbeddingLayer *embedding_layer, Tensor *inputs);
void            embedding_layer_write(EmbeddingLayer *embedding_layer, const char *filename);
#endif