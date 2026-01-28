//#include "../../../include/models/gpt2.h"

#include <stdbool.h>
#include <stdlib.h>
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
    model_gpt_config_print();
}

void model_gpt_init(){
    gpt_model.token_embed_layer = embedding_layer_init(gpt_config.vocab_size,   gpt_config.embed_dim, DTYPE_DOUBLE);
    gpt_model.pos_embed_layer   = embedding_layer_init(gpt_config.context_len,  gpt_config.embed_dim, DTYPE_DOUBLE);
    gpt_model.drop_embed_layer  = dropout_layer_init(gpt_config.drop_rate, false);
    gpt_model.transformer_layers = calloc(gpt_config.n_layers, sizeof(TransformerLayer));
    for(size_t layer_no = 0; layer_no < gpt_config.n_layers; layer_no++){
        gpt_model.transformer_layers[layer_no] = transformer_layer_init(gpt_config.context_len, gpt_config.embed_dim, gpt_config.n_heads, false, true);
    }
    gpt_model.layer_norm        = layer_norm_init(gpt_config.embed_dim);
    gpt_model.out_head_layer    = linear_layer_init(gpt_config.embed_dim, gpt_config.vocab_size, false, false, DTYPE_DOUBLE);
}


Tensor model_gpt_forward(Tensor *input){
    Tensor token_embeddings = embedding_layer_forward(&gpt_model.token_embed_layer, input);     
    tensor_print(&token_embeddings, "token_embeddings");

    Tensor indices = tensor_arange(0, gpt_config.context_len, 1);
    tensor_print(&indices, "indices");
    Tensor pos_embeddings   = embedding_layer_forward(&gpt_model.pos_embed_layer,   &indices);
    tensor_print(&pos_embeddings,   "pos_embeddings");
    //It works till here

    Tensor input_embeddings = tensor_add(&token_embeddings, &pos_embeddings);
    tensor_print(&input_embeddings, "input_embeddings");

    Tensor embeddings = tensor_copy(&input_embeddings);
    tensor_print(&embeddings, "embeddings");
    for(size_t layer_no = 0; layer_no < gpt_config.n_layers; layer_no++){
        Tensor embeddings = transformer_layer_forward(&gpt_model.transformer_layers[layer_no], &embeddings, false);
    }


    dropout_layer_forward(&gpt_model.drop_embed_layer, &input_embeddings);
    tensor_print(&input_embeddings, "input_embeddings (after dropout)");

    //Add transformer block here



    Tensor output = linear_layer_forward(&gpt_model.out_head_layer, &input_embeddings);
    tensor_print(&output, "output");

    Tensor output_norm =  layer_norm_forward(&gpt_model.layer_norm, &output);
    tensor_print(&output_norm, "output_norm");

    return output_norm;
}

void model_gpt_write(){
    embedding_layer_write(&gpt_model.token_embed_layer, "./output/token_embed_layer.csv");
    embedding_layer_write(&gpt_model.pos_embed_layer,   "./output/pos_embed_layer.csv");
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