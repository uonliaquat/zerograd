#include "../../include/layers/embedding.h"

#include <stdlib.h>

EmbeddingLayer embedding_layer_init(size_t rows, size_t cols, size_t seq_len, DataType dtype){
    EmbeddingLayer embedding_layer;
    embedding_layer.weights = tensor_init(NULL, (size_t[]){rows, cols}, 2, dtype, false, true);
    embedding_layer.dtype = dtype;
    embedding_layer.vocab_size = rows;
    embedding_layer.embed_len = cols;
    embedding_layer.seq_len = seq_len;
    return embedding_layer;
}

void embedding_layer_free(EmbeddingLayer *embedding_layer){
    tensor_free(&embedding_layer->weights);
}

Tensor embedding_layer_token_forward(EmbeddingLayer *embedding_layer, Tensor *inputs){
    //This function can be optimized by directly keeping the pointers to rows from weights
    printf("\n==============Token Embedding Layer Forward Pass==================\n");
    Tensor output = tensor_init(
        NULL, 
        (size_t[]){embedding_layer->seq_len,
        embedding_layer->embed_len},
        2,
        embedding_layer->dtype,
        embedding_layer->weights.requires_grad,
        false
    );

    for(size_t i = 0; i < embedding_layer->seq_len; i++){
        int row_id = ((int*)inputs->data)[i];
        //printf("Copying row: %d \n", row_id);
        tensor_copy_row_data(&output, i, &embedding_layer->weights, row_id, embedding_layer->embed_len);
    }
    return output;
}

Tensor embedding_layer_positional_forward(EmbeddingLayer *embedding_layer){
    //This function can be optimized by directly keeping the pointers to rows from weights
    printf("\n==============Position Embedding Layer Forward Pass==================\n");
    Tensor output = tensor_init(
        NULL, 
        (size_t[]){embedding_layer->seq_len,
        embedding_layer->embed_len},
        2,
        embedding_layer->dtype,
        embedding_layer->weights.requires_grad,
        false
    );
    for(size_t i = 0; i < embedding_layer->seq_len; i++){
        int row_id = i;
        //printf("Copying row: %d \n", row_id);
        tensor_copy_row_data(&output, i, &embedding_layer->weights, row_id, embedding_layer->embed_len);
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

