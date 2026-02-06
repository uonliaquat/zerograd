//#include "../../../include/models/gpt2.h"

#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include "../../include/tensor.h"
#include "../../include/utils.h"
#include "../../include/models/gpt.h"
#include "../../include/safetensors.h"



static inline void model_gpt_config_init(GPTConfig *config, 
    const size_t vocab_size, 
    const size_t context_len, 
    const size_t embed_dim, 
    const size_t n_heads, 
    const size_t n_layers, 
    const float drop_rate, 
    const bool qkv_bias, 
    const size_t batch_size,
    const DataType dtype
){
        config->vocab_size  = vocab_size;
        config->context_len = context_len;
        config->embed_dim   = embed_dim;
        config->n_heads     = n_heads;
        config->n_layers    = n_layers;
        config->drop_rate   = drop_rate;
        config->qkv_bias    = qkv_bias;
        config->batch_size  = batch_size;
        config->dtype       = dtype;
}

static inline void model_gpt_workspace_init(GPTModel *model){
    tensor_reset(&model->workspace.indices);
    tensor_reset(&model->workspace.input_embeddings);
    tensor_reset(&model->workspace.position_indices);
}

static inline void model_gpt_workspace_free(GPTWrokspace *workspace){
    tensor_free(&workspace->indices);
    tensor_free(&workspace->position_indices);
    tensor_free(&workspace->input_embeddings);
}


GPTModel model_gpt_init(GPTParams *params, 
    const size_t vocab_size,
    const size_t context_len, 
    const size_t embed_dim, 
    const size_t n_heads, 
    const size_t n_layers, 
    const float drop_rate, 
    const bool qkv_bias, 
    const size_t batch_size,
    const DataType dtype
){
    GPTModel model;
    model_gpt_config_init(&model.config, vocab_size, context_len, embed_dim, n_heads, n_layers, drop_rate, qkv_bias, batch_size, dtype);
    model.params        = params;
    model.wte_layer     = embedding_layer_init(&model.params->wpe, vocab_size,  embed_dim, DTYPE_FP32);
    model.wpe_layer     = embedding_layer_init(&model.params->wte, context_len, embed_dim, DTYPE_FP32);
    for(size_t i = 0; i < n_heads; i++){
        model.h_layer[i] = transformer_layer_init(&model.params->h[i], context_len, embed_dim, n_heads, dtype);
    }
    model_gpt_workspace_init(&model);
    return model;
}


void model_gpt_free(GPTModel *model){
    model_gpt_workspace_free(&model->workspace);
    embedding_layer_free(&model->wte_layer);
    embedding_layer_free(&model->wpe_layer);
    //transformer_block_free(&model->transformer_block);
    // tensor_free(&gpt_model.output);
}

void model_gpt_forward(GPTModel *model, Tensor *input){
    assert(input->ndim == 3);
    embedding_layer_forward(&model->wte_layer, input);
    tensor_print(&model->wte_layer.output, "wte_layer.output");


    tensor_arange_(0, input->shape[input->ndim-1], 1, &model->workspace.position_indices);
    tensor_print(&model->workspace.position_indices, "position_indices (arrange)");

    // tensor_repeat_(&model->workspace.indices, (uint8_t[]){input->shape[1], 1}, &model->workspace.position_indices);
    // tensor_print(&model->workspace.position_indices);

    tensor_unsqueeze_(&model->workspace.position_indices, 0);
    tensor_print(&model->workspace.position_indices, "position_indices (unsqueezed)");

    embedding_layer_forward(&model->wpe_layer, &model->workspace.position_indices);
    tensor_print(&model->wpe_layer.output, "wpe_layer.output");


    tensor_add_(&model->wte_layer.output, &model->wpe_layer.output, &model->workspace.input_embeddings);
    tensor_print(&model->workspace.input_embeddings, "input_embeddings");


    for(size_t i = 0; i < 1; i++){
        transformer_layer_forward(&model->h_layer[i], &model->workspace.input_embeddings, false);
    }

    // transformer_block_forward(&model.transformer_block,  &model.workspace.input_embeddings);
    //transformer_block_print(&model.transformer_block, "Transformer Block");
    // //Tensor embeddings = tensor_copy(&input_embeddings);
    // Tensor *embeddings = calloc(gpt_config.n_layers+1, sizeof(Tensor));
    // embeddings[0] =  tensor_copy(&input_embeddings);
    // tensor_print(&embeddings[0], "embeddings (after copy)");
    // for(size_t layer_no = 1; layer_no <= gpt_config.n_layers; layer_no++){
    //     tensor_print(&embeddings[layer_no-1], "embeddings[layer_no-1]");
    //     embeddings[layer_no] = transformer_layer_forward(&model.transformer_layers[layer_no-1], &embeddings[layer_no-1], false);
    //     tensor_free(&embeddings[layer_no-1]);
    // }


    // dropout_layer_forward(&model.drop_embed_layer, &embeddings);
    // tensor_print(&embeddings[0], "embeddings (after dropout)");




    // Tensor output = linear_layer_forward(&gpt_model.out_head_layer, &embeddings[gpt_config.n_layers]);
    // tensor_print(&output, "model_gpt_forward output");

    // Tensor output_norm =  layer_norm_forward(&gpt_model.layer_norm, &output);
    // tensor_print(&output_norm, "output_norm");
}


void model_gpt_safetensors_write(const char *filename, GPTParams *params){
    Tensor *tensors[500];
    size_t idx = 0;
    tensors[idx++] = &params->wte.weight;
    tensors[idx++] = &params->wpe.weight;
    for(size_t i = 0; i < 12; i++){
        tensors[idx++] = &params->h[i].attn.bias;
        tensors[idx++] = &params->h[i].attn.c_attn.bias;
        tensors[idx++] = &params->h[i].attn.c_attn.weight;
        tensors[idx++] = &params->h[i].attn.c_proj.bias;
        tensors[idx++] = &params->h[i].attn.c_proj.weight;
        tensors[idx++] = &params->h[i].ln_[0].bias;
        tensors[idx++] = &params->h[i].ln_[0].weight;
        tensors[idx++] = &params->h[i].ln_[1].bias;
        tensors[idx++] = &params->h[i].ln_[1].weight;
        tensors[idx++] = &params->h[i].mlp.c_fc.bias;
        tensors[idx++] = &params->h[i].mlp.c_fc.weight;
        tensors[idx++] = &params->h[i].mlp.c_proj.bias;
        tensors[idx++] = &params->h[i].mlp.c_proj.weight;
    }
    tensors[idx++] = &params->ln_f.bias;
    tensors[idx++] = &params->ln_f.weight;
    safetensors_save_model(filename, tensors, idx);
}



// void model_gpt_rand_init(){
//     model_gpt_workspace_init();

//     gpt_model.token_embed_layer = embedding_layer_init(gpt_config.vocab_size,   gpt_config.embed_dim, DTYPE_FP32);
//     gpt_model.pos_embed_layer   = embedding_layer_init(gpt_config.context_len,  gpt_config.embed_dim, DTYPE_FP32);
//     gpt_model.transformer_block = transformer_block_init(gpt_config.context_len, gpt_config.embed_dim, gpt_config.n_heads, gpt_config.n_layers, gpt_config.drop_rate, gpt_config.qkv_bias, true);
//     // gpt_model.drop_embed_layer  = dropout_layer_init(gpt_config.drop_rate, false);
//     // gpt_model.transformer_layers = calloc(gpt_config.n_layers, sizeof(TransformerLayer));
//     // for(size_t layer_no = 0; layer_no < gpt_config.n_layers; layer_no++){
//     //     gpt_model.transformer_layers[layer_no] = transformer_layer_init(gpt_config.context_len, gpt_config.embed_dim, gpt_config.n_heads, false, true);
//     // }
//     // gpt_model.layer_norm        = layer_norm_init(gpt_config.embed_dim);
//     // gpt_model.out_head_layer    = linear_layer_init(gpt_config.embed_dim, gpt_config.vocab_size, false, false, DTYPE_FP32);
// }


