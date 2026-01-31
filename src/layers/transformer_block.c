#include "../../include/utils.h"
#include "../../include/layers/transformer_block.h"
#include<string.h>
TransformerBlock transformer_block_init(size_t n_heads, size_t n_layers, size_t drop_rate, bool qkv_bias, bool decoder){
    TransformerBlock transformer_block;
    transformer_block.n_layers = n_layers;
    transformer_block.decoder = decoder;
    transformer_block.transformer_layers = calloc(n_layers, sizeof(TransformerLayer));
    return transformer_block;
}

void transformer_block_free(TransformerBlock *transformer_block){
    for(size_t layer_no = 0; layer_no < transformer_block->n_layers; layer_no++){
        transformer_layer_free(&transformer_block->transformer_layers[layer_no]);
    }
}

void transformer_block_forward(TransformerBlock *transformer_block, Tensor *input){
    for(size_t layer_no = 0; layer_no < transformer_block->n_layers; layer_no++){
        transformer_layer_forward(&transformer_block->transformer_layers[layer_no], input, transformer_block->decoder);
    }
}

void transformer_block_write(TransformerBlock *transformer_block, const char *base_path){
    for(size_t layer_no = 0; layer_no < transformer_block->n_layers; layer_no++){
        char filename[512] = "/0";
        snprintf(filename, 512, "%stransformer_layer_%zu", base_path, layer_no);
        //create_filename(base_path, layer_name, filename);
        transformer_layer_write(&transformer_block->transformer_layers[layer_no], filename);
    }
}