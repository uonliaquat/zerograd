
#include "../../include/layers/embedding.h"
#include "../../include/utils.h"

#include <assert.h>
#include <stdlib.h>


static inline void embedding_layer_workspace_init(EmbeddingLayerWorkspace *workspace, char *name){ 
    char tmp_name[128] = "\0";
    snprintf(tmp_name, sizeof(tmp_name), "%s.weight", name);
    tensor_reset(&workspace->output, name);
}

static inline void embedding_layer_workspace_free(const EmbeddingLayerWorkspace *workspace){ 
    tensor_free(&workspace->output);
}

static inline void embedding_layer_params_free(const EmbeddingLayerParams *params){ 
    tensor_free(&params->weight);
}


EmbeddingLayer  embedding_layer_init(EmbeddingLayerParams *params, const size_t embed_dim, char *name){
    EmbeddingLayer layer;
    strcpy(layer.name, name);
    layer.embed_dim = embed_dim;
    layer.params = params;

    embedding_layer_workspace_init(&layer.workspace, name);

    return layer;
}

void embedding_layer_free(const EmbeddingLayer *layer){
    embedding_layer_workspace_free(&layer->workspace);
    embedding_layer_params_free(layer->params);
}

void embedding_layer_forward(EmbeddingLayer *layer, Tensor *x){
    //This function can be optimized by directly keeping the pointers to rows from weights
    assert(x->ndim == 3);
    if(layer->workspace.output.size == 0){
        tensor_init_(
            &layer->workspace.output,
            NULL,
            (uint32_t[]){x->shape[1], x->shape[2], layer->embed_dim},
            3,
            DTYPE_FP32,
            NULL
        );
    }
    //tensor_print(&layer->workspace.output, "&layer->workspace.output");
    for(size_t batch_id = 0; batch_id < x->shape[1]; batch_id++){
        for(size_t row_id = 0; row_id < x->shape[2]; row_id++){
            int embed_index = tensor_get_elem(x, (uint32_t[]){0, batch_id, row_id});
            //printf("batch_id: %zu,  row_id: %zu, embed_index: %d\n", batch_id, row_id, embed_index);
            tensor_copy_row_data(&layer->workspace.output, batch_id, row_id, &layer->params->weight, embed_index, layer->embed_dim);
        }
    }
    //tensor_print(&layer->workspace.output, "&layer->workspace.output");
}

void embedding_layer_write(EmbeddingLayer *layer, Tensor **tensors, size_t *tensors_len){
    tensors[(*tensors_len)++] = &layer->workspace.output;
}
