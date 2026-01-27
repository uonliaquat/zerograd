#include "../../include/layers/embedding.h"

#include <stdlib.h>

EmbeddingLayer embedding_layer_init(size_t num_embed, size_t embed_dim, DataType dtype){
    EmbeddingLayer embedding_layer;
    embedding_layer.weights = tensor_init(NULL, (size_t[]){num_embed, embed_dim}, 2, dtype, false, true);
    embedding_layer.dtype = dtype;
    embedding_layer.num_embed = num_embed;
    embedding_layer.embed_dim = embed_dim;
    return embedding_layer;
}

void embedding_layer_free(EmbeddingLayer *embedding_layer){
    tensor_free(&embedding_layer->weights);
}

Tensor embedding_layer_forward(EmbeddingLayer *embedding_layer, Tensor *input){
    //This function can be optimized by directly keeping the pointers to rows from weights
    printf("\n==============Token Embedding Layer Forward Pass==================\n");
    Tensor output = tensor_init(
        NULL, 
        (size_t[]){input->shape[0], input->shape[1], embedding_layer->embed_dim},
        3,
        embedding_layer->dtype,
        embedding_layer->weights.requires_grad,
        false
    );

    for(size_t i = 0; i < input->shape[0]; i++){
        for(size_t j = 0; j < input->shape[1]; j++){
            int embed_index = tensor_get_elem(input, (size_t[]){i, j});
            tensor_copy_row_data(&output, i, j, &embedding_layer->weights, embed_index, embedding_layer->embed_dim);
        }
    }
    return output;
}


void embedding_layer_write(EmbeddingLayer *embedding_layer, const char *filename){
    FILE *fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    fprintf(fptr, "Weights:\n");
    tensor_write_fp(&embedding_layer->weights, fptr);
    //fprintf(fptr, "Output:\n");
    //tensor_write(&embedding_layer->output, fptr);
    fclose(fptr);
}

