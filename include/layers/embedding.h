#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include <stdio.h>
#include "../tensor.h"

#define MAX_SEQUENCE_LENGTH 100

typedef struct EmbeddingLayer{
    Tensor      *weights;
    DataType    dtype;
    size_t      num_embed;
    size_t      embed_dim;
    Tensor      output;
} EmbeddingLayer;


EmbeddingLayer  embedding_layer_init(Tensor *params);
EmbeddingLayer  embedding_layer_rand_init(const size_t num_embed, const size_t embed_dim, const DataType dtype);
void            embedding_layer_free(const EmbeddingLayer *embedding_layer);
void            embedding_layer_forward(EmbeddingLayer *embedding_layer, const Tensor *input);
void            embedding_layer_print(const EmbeddingLayer *embedding_layer, const char *heading);
void            embedding_layer_write(const EmbeddingLayer *embedding_layer, const char *filename);
void            embedding_layer_write_fp(const EmbeddingLayer *embedding_layer, FILE *fptr);
#endif