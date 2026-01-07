#include "../../include/layers/embedding.h"

#include <stdlib.h>

EmbeddingLayer embedding_layer_init(size_t vocab_size, size_t embed_len, size_t seq_len, DataType dtype){
    EmbeddingLayer embedding_layer;
    embedding_layer.weights = tensor_init(NULL, (size_t[]){vocab_size, embed_len}, 2, dtype, false, true);
    embedding_layer.output = tensor_init(NULL, (size_t[]){seq_len, embed_len}, 2, dtype, false, false);
    embedding_layer.dtype = dtype;
    embedding_layer.vocab_size = vocab_size;
    embedding_layer.embed_len = embed_len;
    embedding_layer.seq_len = seq_len;
    return embedding_layer;
}

void embedding_layer_free(EmbeddingLayer *embedding_layer){
    tensor_free(&embedding_layer->weights);
    tensor_free(&embedding_layer->output);
}

void embedding_layer_forward(EmbeddingLayer *embedding_layer, Tensor *inputs){
    //This function can be optimized by directly keeping the pointers to rows from weights
    printf("\n==============Embedding Layer Forward Pass==================\n");
    for(size_t i = 0; i < embedding_layer->seq_len; i++){
        int row_id = ((int*)inputs->data)[i];
        //printf("Copying row: %d \n", row_id);
        tensor_copy_row_data(&embedding_layer->output, i, &embedding_layer->weights, row_id, embedding_layer->embed_len);
    }

}

void embedding_layer_write(EmbeddingLayer *embedding_layer, const char *filename){
    FILE *fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    fprintf(fptr, "Weights:\n");
    tensor_write(&embedding_layer->weights, fptr);
    fprintf(fptr, "Output:\n");
    tensor_write(&embedding_layer->output, fptr);
    fclose(fptr);
}

