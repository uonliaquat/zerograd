#include "../../include/utils.h"
#include "../../include/layers/embedding.h"

#include <stdlib.h>

EmbeddingLayer embedding_layer_init(const size_t num_embed, const size_t embed_dim, const DataType dtype){
    EmbeddingLayer embedding_layer;
    embedding_layer.weights     = tensor_init(NULL, (size_t[]){num_embed, embed_dim}, 2, dtype, false, true);
    embedding_layer.dtype       = dtype;
    embedding_layer.num_embed   = num_embed;
    embedding_layer.embed_dim   = embed_dim;
    embedding_layer.output      = (Tensor){0};
    return embedding_layer;
}

void embedding_layer_free(const EmbeddingLayer *embedding_layer){
    tensor_free(&embedding_layer->weights);
    tensor_free(&embedding_layer->output);
}

void embedding_layer_forward(EmbeddingLayer *embedding_layer, const Tensor *input){
    //This function can be optimized by directly keeping the pointers to rows from weights

    if(embedding_layer->output.size == 0){
        printf("Creating output tensor\n");
        embedding_layer->output = tensor_init(
            NULL, 
            (size_t[]){input->shape[1], input->shape[2], embedding_layer->embed_dim},
            3,
            embedding_layer->dtype,
            embedding_layer->weights.requires_grad,
            false
        );
    }
    
    for(size_t i = 0; i < input->shape[1]; i++){
        for(size_t j = 0; j < input->shape[2]; j++){
            int embed_index = tensor_get_elem(input, (size_t[]){0, i, j});
            tensor_copy_row_data(&embedding_layer->output, i, j, &embedding_layer->weights, embed_index, embedding_layer->embed_dim);
        }
    }
}

void embedding_layer_print(const EmbeddingLayer *embedding_layer, const char *heading){
    print_centered_heading(heading);
    tensor_print(&embedding_layer->weights, "embedding_layer->weights");
    //if(embedding_layer->output.size != 0){
    tensor_print(&embedding_layer->output,  "embedding_layer->output");
    //}
}


void embedding_layer_write(const EmbeddingLayer *embedding_layer, const char *filename){
    FILE *fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    
    fprintf(fptr, "Weights\n");
    tensor_write_fp(&embedding_layer->weights, fptr);
    fprintf(fptr, "Output\n");
    tensor_write_fp(&embedding_layer->output, fptr);
    fclose(fptr);
}

void embedding_layer_write_fp(const EmbeddingLayer *embedding_layer, FILE *fptr){
    fprintf(fptr, "Weights\n");
    tensor_write_fp(&embedding_layer->weights, fptr);
    fprintf(fptr, "Output\n");
    tensor_write_fp(&embedding_layer->output, fptr);
}

