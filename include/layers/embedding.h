#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include <stdio.h>
#include "../tensor.h"

#define MAX_SEQUENCE_LENGTH 100

typedef struct EmbeddingLayer{
    Tensor weights;
    DataType dtype;
    size_t num_embed;
    size_t embed_dim;
} EmbeddingLayer;


EmbeddingLayer  embedding_layer_init(size_t num_embed, size_t embed_dim, DataType dtype);
void            embedding_layer_free(EmbeddingLayer *embedding_layer);
Tensor          embedding_layer_token_forward(EmbeddingLayer *embedding_layer, Tensor *inputs);
Tensor          embedding_layer_positional_forward(EmbeddingLayer *embedding_layer);
void            embedding_layer_write(EmbeddingLayer *embedding_layer, const char *filename);
#endif