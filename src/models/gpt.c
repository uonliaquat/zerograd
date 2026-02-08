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
    const DataType dtype,
    char *name
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
        memset(config->name, 0, sizeof(config->name));
        strcpy(config->name, name);
}

static inline void model_gpt_workspace_init(GPTModel *model, const size_t n_layers, char *name){

    char l_name[256] = "\0";  

    snprintf(l_name, sizeof(l_name), "%s.indices", name);
    tensor_reset(&model->workspace.indices, l_name);

    for(size_t i = 0; i < n_layers+1; i++){
        snprintf(l_name, sizeof(l_name), "%s.embeddings.%zu", name, i);
        tensor_reset(&model->workspace.embeddings[i], l_name);
    }
    
    snprintf(l_name, sizeof(l_name), "%s.position.indices", name);
    tensor_reset(&model->workspace.position_indices, l_name);

    snprintf(l_name, sizeof(l_name), "%s.next_token_prob_dist", name);
    tensor_init_(
        &model->workspace.next_token_prob_dist,
        NULL, 
        (uint32_t[]){1, 1, model->config.vocab_size},
        3, 
        model->config.dtype,
        l_name
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
    const DataType dtype,
    char *name
){
    GPTModel model;
    model_gpt_config_init(&model.config, vocab_size, context_len, embed_dim, n_heads, n_layers, drop_rate, qkv_bias, batch_size, dtype, name);
    model.params            = params;

    char l_name[256] = "\0";
    snprintf(l_name, sizeof(l_name), "%s.wte", name);
    model.wte_layer         = embedding_layer_init(&model.params->wte, vocab_size,  embed_dim, DTYPE_FP32, l_name);

    snprintf(l_name, sizeof(l_name), "%s.wpe", name);
    model.wpe_layer         = embedding_layer_init(&model.params->wpe, context_len, embed_dim, DTYPE_FP32, l_name);
    for(size_t i = 0; i < n_heads; i++){
        char l_name[128] = "\0";
        snprintf(l_name, sizeof(l_name), "%s.h.%zu", name, i);
        model.h_layer[i] = transformer_layer_init(&model.params->h[i], context_len, embed_dim, n_heads, true, dtype, l_name);
    }
    snprintf(l_name, sizeof(l_name), "%s.ln_f", name);
    model.ln_f_layer        = layer_norm_init(&params->ln_f, dtype, l_name);
    
    snprintf(l_name, sizeof(l_name), "%s.head", name);
    model.head_layer    = linear_layer_init(&params->head, dtype, l_name);

    snprintf(l_name, sizeof(l_name), "%s.output", name);
    tensor_reset(&model.output, l_name);

    model_gpt_workspace_init(&model, n_layers, name);
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

    // size_t next_token_index = 5;
    size_t max_itrs = 10;
    printf("max_itrs: %zu\n", max_itrs);
    for(size_t itr = 0; itr < max_itrs; itr++){
        printf("itr: %zu\n", itr);
        tensor_print(input, "input embeddings");

        embedding_layer_forward(&model->wte_layer, input);

        if(itr == 0){
            tensor_arange_(0, input->shape[input->ndim-1], 1, &model->workspace.position_indices);

            tensor_unsqueeze_(&model->workspace.position_indices, 0);
        }

        embedding_layer_forward(&model->wpe_layer, &model->workspace.position_indices);


        tensor_add_(&model->wte_layer.output, &model->wpe_layer.output, &model->workspace.embeddings[0]);

        for(size_t i = 0; i < model->config.n_layers; i++){
            transformer_layer_forward(&model->h_layer[i], &model->workspace.embeddings[i]);
            tensor_copy_(&model->h_layer[i].output, &model->workspace.embeddings[i+1]);
        }

        layer_norm_forward(&model->ln_f_layer, &model->workspace.embeddings[model->config.n_layers]);
        linear_layer_forward(&model->head_layer, &model->ln_f_layer.output);
        tensor_softmax_(&model->head_layer.output, 1,  &model->output);

        tensor_copy_row_data(&model->workspace.next_token_prob_dist, 0, 0, &model->output, model->output.shape[model->output.ndim-2]-1, model->config.vocab_size);

        size_t next_token_id = get_next_token_id(&model->workspace.next_token_prob_dist);
        printf("\n Token | next_token_id: %zu, token: %s\n", next_token_id, vocab.tokens[next_token_id].token);

        // Tensor new_input = tensor_init(
        //     NULL,
        //     input->shape,
        //     input->ndim,
        //     input->dtype,
        //     input->name
        // );
        // memcpy(new_input.data, &(input->data[1]), input->elem_size * input->size-1);
        // ((int*)new_input.data)[51] = next_token_id;

        // tensor_free(input);
        // input = &new_input;
        // ((float*)(input->data))[next_token_index++] = next_token_id;
        
        // tensor_copy_row_data(input, 0, next_token_index, &model->wte_layer.params->weight, next_token_id, model->config.embed_dim);

    }
}


void model_gpt_write(GPTModel *model, const char *filename){
    Tensor *tensors[200];
    size_t tensors_len = 0;
    embedding_layer_write(&model->wte_layer, tensors, &tensors_len);
    embedding_layer_write(&model->wpe_layer, tensors, &tensors_len);

    for(size_t i = 0; i < model->config.n_layers; i++){
        tensors[tensors_len++] = &model->workspace.embeddings[i];
        transformer_layer_write(&model->h_layer[i], tensors, &tensors_len);
    }

    layer_norm_write(&model->ln_f_layer, tensors, &tensors_len);
    linear_layer_write(&model->head_layer, tensors, &tensors_len);
    
    tensors[tensors_len++] = &model->output;
    tensors[tensors_len++] = &model->workspace.next_token_prob_dist;
    
    strcpy(model->params->head.weight.name, "head.weight");
    tensors[tensors_len++] = &model->head_layer.params->weight;

    printf("\n\ntensors_len: %zu\n", tensors_len);
    safetensors_save_model(filename, tensors, tensors_len);
}
// void model_gpt_safetensors_write(const char *filename, GPTParams *params){
//     Tensor *tensors[500];
//     size_t idx = 0;
//     tensors[idx++] = &params->wte.weight;
//     tensors[idx++] = &params->wpe.weight;
//     for(size_t i = 0; i < 12; i++){
//         tensors[idx++] = &params->h[i].attn.bias;
//         tensors[idx++] = &params->h[i].c_attn.bias;
//         tensors[idx++] = &params->h[i].c_attn.weight;
//         tensors[idx++] = &params->h[i].attn.c_proj.bias;
//         tensors[idx++] = &params->h[i].attn.c_proj.weight;
//         tensors[idx++] = &params->h[i].ln_[0].bias;
//         tensors[idx++] = &params->h[i].ln_[0].weight;
//         tensors[idx++] = &params->h[i].ln_[1].bias;
//         tensors[idx++] = &params->h[i].ln_[1].weight;
//         tensors[idx++] = &params->h[i].mlp.c_fc.bias;
//         tensors[idx++] = &params->h[i].mlp.c_fc.weight;
//         tensors[idx++] = &params->h[i].mlp.c_proj.bias;
//         tensors[idx++] = &params->h[i].mlp.c_proj.weight;
//     }
//     tensors[idx++] = &params->ln_f.bias;
//     tensors[idx++] = &params->ln_f.weight;
//     safetensors_save_model(filename, tensors, idx);
// }



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


