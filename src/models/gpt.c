//#include "../../../include/models/gpt2.h"

#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include "../../include/tokenizer.h"
#include "../../include/tensor.h"
#include "../../include/utils.h"
#include "../../include/models/gpt.h"
#include "../../include/safetensors.h"
#include <float.h>



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

static inline void model_gpt_workspace_init(GPTModel *model, const size_t n_layers){
    tensor_reset(&model->workspace.indices);
    for(size_t i = 0; i < n_layers+1; i++){
        tensor_reset(&model->workspace.embeddings[i]);
    }
    tensor_reset(&model->workspace.position_indices);
    tensor_init_(
        &model->workspace.next_token_prob_dist,
        NULL, 
        (uint32_t[]){1, 1, model->config.vocab_size},
        3, 
        model->config.dtype,
        "next_token_prob_dist"
    );
}

static inline void model_gpt_workspace_free(GPTWrokspace *workspace, const size_t n_layers){
    tensor_free(&workspace->indices);
    tensor_free(&workspace->position_indices);
    for(size_t i = 0; i < n_layers+1; i++){
        tensor_free(&workspace->embeddings[i]);
    }
    tensor_free(&workspace->next_token_prob_dist);
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
    model.params            = params;
    model.wte_layer         = embedding_layer_init(&model.params->wpe, vocab_size,  embed_dim, DTYPE_FP32);
    model.wpe_layer         = embedding_layer_init(&model.params->wte, context_len, embed_dim, DTYPE_FP32);
    for(size_t i = 0; i < n_heads; i++){
        model.h_layer[i] = transformer_layer_init(&model.params->h[i], context_len, embed_dim, n_heads, true, dtype);
    }
    model.ln_f_layer        = layer_norm_init(&params->ln_f, dtype);
    model.out_proj_layer    = linear_layer_init(&params->out_proj, dtype);
    tensor_reset(&model.output);
    model_gpt_workspace_init(&model, n_layers);
    return model;
}


void model_gpt_free(GPTModel *model){
    model_gpt_workspace_free(&model->workspace, model->config.n_layers);
    embedding_layer_free(&model->wte_layer);
    embedding_layer_free(&model->wpe_layer);
    for(size_t i = 0; i < model->config.n_layers; i++){
        transformer_layer_free(&model->h_layer[i]);
    }
    layer_norm_free(&model->ln_f_layer);
    tensor_free(&model->output);
}

static inline size_t get_next_token_id(Tensor * next_token_prob_dist){
    float max_prob = -1;
    size_t token_id = 0;
    for(size_t i = 0; i < next_token_prob_dist->size; i++){
        float prob = ((float*)next_token_prob_dist->data)[i];
        if(prob > max_prob){
            max_prob = prob;
            token_id = i;
        } 
    }
    return token_id;
}

void model_gpt_forward(GPTModel *model, Tensor *input){
    assert(input->ndim == 3);

    Vocab vocab = tokenizer_read_vocab("/Users/uonliaquat/workspace/zerograd/python/gpt2_vocab.txt");
    // for(size_t i = 0; i < vocab.len; i++){
    //     printf("%s\n", vocab.tokens[i].token);
    // }

    size_t next_token_index = 5;
    size_t max_itrs = input->shape[input->ndim - 1];
    printf("max_itrs: %zu\n", max_itrs);
    for(size_t itr = 0; itr < max_itrs; itr++){
        printf("itr: %zu\n", itr);
        tensor_print(input, "input embeddings");
        // uint32_t shape_i = input->shape[0];
        // uint32_t shape_j = input->shape[1];
        // uint32_t shape_k = input->shape[2];
        // printf("[\n");
        // for(size_t i = 0; i < shape_i; i++){
        //     for(size_t j = 0; j < shape_j; j++){
        //         printf("    [ ");
        //         for(size_t k = 0; k < shape_k; k++){
        //             float elem = tensor_get_elem(input, (uint32_t[]){i, j, k});
        //             if(input->dtype == DTYPE_FP32){ 
        //                 if(elem == -FLT_MAX){
        //                     printf("%s ", "-INF");
        //                 }
        //                 else{
        //                     printf("%2.2f ", elem);
        //                 }
        //             }
        //             else if(input->dtype == DTYPE_INT32) printf("%d   ", (int)elem);
        //         }
        //         printf(" ]\n");
        //     }
        //     // if(i <= shape_i - 2) printf("\n\n");
        // }
        // printf("]\n");

        embedding_layer_forward(&model->wte_layer, input);
        tensor_print(&model->wte_layer.output, "wte_layer.output");


        if(itr == 0){
            tensor_arange_(0, input->shape[input->ndim-1], 1, &model->workspace.position_indices);
            tensor_print(&model->workspace.position_indices, "position_indices (arrange)");

            tensor_unsqueeze_(&model->workspace.position_indices, 0);
            tensor_print(&model->workspace.position_indices, "position_indices (unsqueezed)");
        }

        embedding_layer_forward(&model->wpe_layer, &model->workspace.position_indices);
        tensor_print(&model->wpe_layer.output, "wpe_layer.output");


        tensor_add_(&model->wte_layer.output, &model->wpe_layer.output, &model->workspace.embeddings[0]);
        tensor_print(&model->workspace.embeddings[0], "input_embeddings");

        for(size_t i = 0; i < model->config.n_layers; i++){
            if(tensor_isnan(&model->workspace.embeddings[i])){
                printf("\n\n\nexiting due to nan values in , &model->workspace.embeddings[i]\n");
                exit(1);
            }
            transformer_layer_forward(&model->h_layer[i], &model->workspace.embeddings[i]);
            tensor_copy_(&model->h_layer[i].output, &model->workspace.embeddings[i+1]);

        }
        layer_norm_forward(&model->ln_f_layer, &model->workspace.embeddings[model->config.n_layers]);
        tensor_print(&model->ln_f_layer.output, "gpt layer_norm (output)");

        linear_layer_forward(&model->out_proj_layer, &model->ln_f_layer.output);
        tensor_print(&model->out_proj_layer.output, "out_proj_layer (output)");

        tensor_softmax_(&model->out_proj_layer.output, 1,  &model->output);

        tensor_print(&model->output, "model->output (softmax)");
        printf("row to copy %u  \n", model->output.shape[model->output.ndim-2]-1);
        tensor_copy_row_data(&model->workspace.next_token_prob_dist, 0, 0, &model->output, model->output.shape[model->output.ndim-2]-1, model->config.vocab_size);
        tensor_print(&model->workspace.next_token_prob_dist, "next_token_prob_dist");
        // //tensor_print(&output_token_tensor, "GPT Model Output");

        size_t next_token_id = get_next_token_id(&model->workspace.next_token_prob_dist);
        printf("\n Token | next_token_id: %zu, token: %s\n", next_token_id, vocab.tokens[next_token_id].token);

        ((float*)(input->data))[next_token_index++] = next_token_id;
        
        // tensor_copy_row_data(input, 0, next_token_index, &model->wte_layer.params->weight, next_token_id, model->config.embed_dim);

    }
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


