#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include <stdio.h>
#include "../tensor.h"

#define MAX_SEQUENCE_LENGTH 100


typedef struct EmbeddingLayerWorkspace{
    Tensor output;
} EmbeddingLayerWorkspace;

typedef struct EmbeddingLayerParams{
    Tensor weight;
} EmbeddingLayerParams;


typedef struct EmbeddingLayer{
    char   name[128];
    size_t embed_dim;
    EmbeddingLayerParams *params;
    EmbeddingLayerWorkspace workspace;
    // size_t      num_embed;
    // size_t      embed_dim;
} EmbeddingLayer;




EmbeddingLayer  embedding_layer_init(EmbeddingLayerParams *params, const size_t embed_dim, char *name);
void            embedding_layer_free(const EmbeddingLayer *layer);
void            embedding_layer_forward(EmbeddingLayer *layer, Tensor *x);
void            embedding_layer_write(EmbeddingLayer *layer, Tensor **tensors, size_t *tensors_len);
#endif