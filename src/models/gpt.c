//#include "../../../include/models/gpt2.h"

#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include "../../include/utils.h"
#include "../../include/models/gpt.h"

static GPTConfig gpt_config;
static GPTModel gpt_model;


void model_gpt_config_init(size_t vocab_size, size_t context_len, size_t embed_dim, size_t n_heads, size_t n_layers, double drop_rate, bool qkv_bias){
    gpt_config.vocab_size   = vocab_size;
    gpt_config.context_len  = context_len;
    gpt_config.embed_dim    = embed_dim;
    gpt_config.n_heads      = n_heads;
    gpt_config.n_layers     = n_layers;
    gpt_config.drop_rate    = drop_rate;
    gpt_config.qkv_bias     = qkv_bias;
}

static inline void model_gpt_workspace_init(){
    gpt_model.workspace.indices             = (Tensor){0};
    gpt_model.workspace.input_embeddings    = (Tensor){0};
    gpt_model.workspace.position_indices   = (Tensor){0};
}

static inline void model_gpt_workspace_free(){
    tensor_free(&gpt_model.workspace.indices);
    tensor_free(&gpt_model.workspace.input_embeddings);
    tensor_free(&gpt_model.workspace.position_indices);
}

void model_gpt_init(){
    model_gpt_workspace_init();

    gpt_model.token_embed_layer = embedding_layer_init(gpt_config.vocab_size,   gpt_config.embed_dim, DTYPE_DOUBLE);
    gpt_model.pos_embed_layer   = embedding_layer_init(gpt_config.context_len,  gpt_config.embed_dim, DTYPE_DOUBLE);
    gpt_model.transformer_block = transformer_block_init(gpt_config.context_len, gpt_config.embed_dim, gpt_config.n_heads, gpt_config.n_layers, gpt_config.drop_rate, gpt_config.qkv_bias, true);
    // gpt_model.drop_embed_layer  = dropout_layer_init(gpt_config.drop_rate, false);
    // gpt_model.transformer_layers = calloc(gpt_config.n_layers, sizeof(TransformerLayer));
    // for(size_t layer_no = 0; layer_no < gpt_config.n_layers; layer_no++){
    //     gpt_model.transformer_layers[layer_no] = transformer_layer_init(gpt_config.context_len, gpt_config.embed_dim, gpt_config.n_heads, false, true);
    // }
    // gpt_model.layer_norm        = layer_norm_init(gpt_config.embed_dim);
    // gpt_model.out_head_layer    = linear_layer_init(gpt_config.embed_dim, gpt_config.vocab_size, false, false, DTYPE_DOUBLE);
}

void model_gpt_free(){
    model_gpt_workspace_free();
    embedding_layer_free(&gpt_model.token_embed_layer);
    embedding_layer_free(&gpt_model.pos_embed_layer);
    transformer_block_free(&gpt_model.transformer_block);
    // tensor_free(&gpt_model.output);
}


void model_gpt_forward(Tensor *input){
    assert(input->ndim == 3);
    embedding_layer_forward(&gpt_model.token_embed_layer, input);  
    embedding_layer_print(&gpt_model.token_embed_layer, "Token Embedding Layer");

    tensor_arange_(0, gpt_config.context_len, 1, &gpt_model.workspace.indices);
    tensor_print(&gpt_model.workspace.indices, "indices");

    tensor_repeat_(&gpt_model.workspace.indices, (size_t[]){input->shape[1], 1}, &gpt_model.workspace.position_indices);
    tensor_print(&gpt_model.workspace.position_indices, "position_indices");

    tensor_unsqueeze_(&gpt_model.workspace.position_indices, 0);
    tensor_print(&gpt_model.workspace.position_indices, "position_indices");

    embedding_layer_forward(&gpt_model.pos_embed_layer, &gpt_model.workspace.position_indices);
    embedding_layer_print(&gpt_model.pos_embed_layer,   "Pos Embedding Layer");



    tensor_add_(&gpt_model.token_embed_layer.output, &gpt_model.pos_embed_layer.output, &gpt_model.workspace.input_embeddings);
    tensor_print(&gpt_model.workspace.input_embeddings, "input_embeddings");

    transformer_block_forward(&gpt_model.transformer_block,  &gpt_model.workspace.input_embeddings);
    //transformer_block_print(&gpt_model.transformer_block, "Transformer Block");
    // //Tensor embeddings = tensor_copy(&input_embeddings);
    // Tensor *embeddings = calloc(gpt_config.n_layers+1, sizeof(Tensor));
    // embeddings[0] =  tensor_copy(&input_embeddings);
    // tensor_print(&embeddings[0], "embeddings (after copy)");
    // for(size_t layer_no = 1; layer_no <= gpt_config.n_layers; layer_no++){
    //     tensor_print(&embeddings[layer_no-1], "embeddings[layer_no-1]");
    //     embeddings[layer_no] = transformer_layer_forward(&gpt_model.transformer_layers[layer_no-1], &embeddings[layer_no-1], false);
    //     tensor_free(&embeddings[layer_no-1]);
    // }


    // dropout_layer_forward(&gpt_model.drop_embed_layer, &embeddings);
    // tensor_print(&embeddings[0], "embeddings (after dropout)");




    // Tensor output = linear_layer_forward(&gpt_model.out_head_layer, &embeddings[gpt_config.n_layers]);
    // tensor_print(&output, "model_gpt_forward output");

    // Tensor output_norm =  layer_norm_forward(&gpt_model.layer_norm, &output);
    // tensor_print(&output_norm, "output_norm");
}

void model_gpt_write(const char *base_path){
    char filename[512] = "\0";
    create_filename(base_path, "gpt_model.token_embed_layer.csv", filename);
    embedding_layer_write(&gpt_model.token_embed_layer,  filename);

    create_filename(base_path, "gpt_model.pos_embed_layer.csv", filename);
    embedding_layer_write(&gpt_model.pos_embed_layer,    filename);

    create_filename(base_path, "gpt_model.workspace.position_indices.csv", filename);
    tensor_write(&gpt_model.workspace.position_indices, filename);

    create_filename(base_path, "gpt_model.workspace.input_embeddings.csv", filename);
    tensor_write(&gpt_model.workspace.input_embeddings,  filename);


    transformer_block_write(&gpt_model.transformer_block, base_path);
}

void model_gpt_config_print(){
    printf("*************** GPT Config **************\n");
    printf("vocab_size:     %zu\n", gpt_config.vocab_size);
    printf("context_len:    %zu\n", gpt_config.context_len);
    printf("embed_dim:      %zu\n", gpt_config.embed_dim);
    printf("n_heads:        %zu\n", gpt_config.n_heads);
    printf("n_layers:       %zu\n", gpt_config.n_layers);
    printf("drop_rate:      %.2f\n", gpt_config.drop_rate);
    printf("qkv_bias:       %s\n", gpt_config.qkv_bias ? "true": "false");
}