
#include "../../include/layers/embedding.h"
#include "../../include/utils.h"

#include <assert.h>
#include <stdlib.h>


EmbeddingLayer embedding_layer_init(EmbeddingLayerParams *params, const size_t num_embed, const size_t embed_dim, const DataType dtype, char *name){
    EmbeddingLayer embedding_layer;
    embedding_layer.params      = params;
    embedding_layer.dtype       = dtype;
    embedding_layer.num_embed   = num_embed;
    embedding_layer.embed_dim   = embed_dim;
    memset(embedding_layer.name, 0, sizeof(embedding_layer.name));
    strcpy(embedding_layer.name, name);

    char l_name[256] = "\0";
    snprintf(l_name, sizeof(l_name), "%s.output", name);
    tensor_reset(&embedding_layer.output, l_name);
    return embedding_layer;
}

// EmbeddingLayer embedding_layer_rand_init(const size_t num_embed, const size_t embed_dim, const DataType dtype){
//     EmbeddingLayer embedding_layer;
//     embedding_layer.weights     = tensor_init(NULL, (uint32_t[]){num_embed, embed_dim}, 2, dtype, NULL);
//     embedding_layer.dtype       = dtype;
//     embedding_layer.num_embed   = num_embed;
//     embedding_layer.embed_dim   = embed_dim;
//     embedding_layer.output      = (Tensor){0};
//     return embedding_layer;
// }

void embedding_layer_free(const EmbeddingLayer *embedding_layer){
    embedding_layer_params_free(embedding_layer->params);
    tensor_free(&embedding_layer->output);
}

void embedding_layer_params_free(const EmbeddingLayerParams *params){
    tensor_free(&params->weight);
}

void embedding_layer_forward(EmbeddingLayer *embedding_layer, const Tensor *input){
    //This function can be optimized by directly keeping the pointers to rows from weights
    assert(input->ndim == 3);
    if(embedding_layer->output.size == 0){
        tensor_init_(
            &embedding_layer->output,
            NULL,
            (uint32_t[]){input->shape[1], input->shape[2], embedding_layer->embed_dim},
            3,
            embedding_layer->dtype,
            NULL
        );
        //tensor_print(embedding_layer->output , "embedding_layer->output ");
    }
    for(size_t batch_id = 0; batch_id < input->shape[1]; batch_id++){
        for(size_t row_id = 0; row_id < input->shape[2]; row_id++){
            int embed_index = tensor_get_elem(input, (uint32_t[]){0, batch_id, row_id});
            //printf("batch_id: %zu,  row_id: %zu, embed_index: %d\n", batch_id, row_id, embed_index);
            tensor_copy_row_data(&embedding_layer->output, batch_id, row_id, &embedding_layer->params->weight, embed_index, embedding_layer->embed_dim);
        }
    }
}

void embedding_layer_write(EmbeddingLayer *embedding_layer, Tensor **tensors, size_t *tensors_len){
    tensors[(*tensors_len)++] = &embedding_layer->output;
}

// void embedding_layer_print(const EmbeddingLayer *embedding_layer, const char *heading){
//     print_centered_heading(heading);
//     tensor_print(&embedding_layer->weights, "embedding_layer->weights");
//     //if(embedding_layer->output.size != 0){
//     tensor_print(&embedding_layer->output,  "embedding_layer->output");
//     //}
// }


// void embedding_layer_write(const EmbeddingLayer *embedding_layer, const char *filename){
//     FILE *fptr = fopen(filename, "w");
//     if(fptr == NULL){
//         printf("Error opening file %s\n", filename);
//         exit(1);
//     }
    
//     fprintf(fptr, "Weights\n");
//     tensor_write_fp(&embedding_layer->weights, fptr);
//     fprintf(fptr, "Output\n");
//     tensor_write_fp(&embedding_layer->output, fptr);
//     fclose(fptr);
// }

// void embedding_layer_write_fp(const EmbeddingLayer *embedding_layer, FILE *fptr){
//     fprintf(fptr, "Weights\n");
//     tensor_write_fp(&embedding_layer->weights, fptr);
//     fprintf(fptr, "Output\n");
//     tensor_write_fp(&embedding_layer->output, fptr);
// }

