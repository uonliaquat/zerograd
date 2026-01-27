//#include "../../../include/models/gpt2.h"

#include "../../include/models/gpt.h"
#include "../../include/tensor.h"

static GPTConfig gpt_config;
static GPTModel gpt_model;

void model_gpt_init_config(size_t vocab_size, size_t context_len, size_t embed_dim, size_t n_heads, size_t n_layers, double drop_rate, bool qkv_bias){
    gpt_config.vocab_size = vocab_size;
    gpt_config.context_len = context_len;
    gpt_config.embed_dim = embed_dim;
    gpt_config.n_heads = n_heads;
    gpt_config.n_layers = n_layers;
    gpt_config.drop_rate = drop_rate;
    gpt_config.qkv_bias = qkv_bias;
}
void model_gpt_init(){
    gpt_model.token_embed_layer = embedding_layer_init(gpt_config.vocab_size,   gpt_config.embed_dim, DTYPE_DOUBLE);
    gpt_model.pos_embed_layer   = embedding_layer_init(gpt_config.context_len,  gpt_config.embed_dim, DTYPE_DOUBLE);
}

void model_gpt_forward(Tensor *input){
    Tensor token_embeddings = embedding_layer_forward(&gpt_model.token_embed_layer, input);     
    tensor_print(&token_embeddings, "token_embeddings");
    Tensor indices = tensor_arange(0, gpt_config.context_len, 1);
    tensor_print(&indices, "indices");
    Tensor pos_embeddings   = embedding_layer_forward(&gpt_model.pos_embed_layer,   &indices);
    tensor_print(&pos_embeddings,   "pos_embeddings");
}

void model_gpt_write(){
    embedding_layer_write(&gpt_model.token_embed_layer, "./output/token_embed_layer.csv");
    embedding_layer_write(&gpt_model.pos_embed_layer,   "./output/pos_embed_layer.csv");
}
