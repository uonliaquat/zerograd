//#include "../../../include/models/gpt2.h"

#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include "../../../include/tensor.h"
#include "../../../include/layers/multi_head_attention.h"
#include "../../../include/utils.h"
#include "../../../include/models/gpt2/gpt.h"
#include "../../../include/safetensors.h"

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

        assert(config->embed_dim % config->n_heads == 0);
}

static inline void model_gpt_workspace_init(GPTModel *model, char *name){

    char tmp_name[128] = "\0";  

    snprintf(tmp_name, sizeof(tmp_name), "indices");
    tensor_reset(&model->workspace.indices, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "pos.indices");
    tensor_reset(&model->workspace.pos_indices, tmp_name);

    for(size_t h = 0; h < model->config.n_layers + 1; h++){
        snprintf(tmp_name, sizeof(tmp_name), "h.%zu.input_embedding", h);
        tensor_reset(&model->workspace.embeddings[h], tmp_name);
    }
    

    snprintf(tmp_name, sizeof(tmp_name), "next_token_prob_dist");
    tensor_init_(
        &model->workspace.next_token_prob_dist,
        NULL, 
        (uint32_t[]){1, 1, model->config.vocab_size},
        3, 
        model->config.dtype,
        tmp_name
    );

    snprintf(tmp_name, sizeof(tmp_name), "probs");
    tensor_reset(&model->workspace.output, tmp_name);
}

static inline void model_gpt_workspace_free(GPTWrokspace *workspace, const size_t n_layers){
    tensor_free(&workspace->indices);
    tensor_free(&workspace->pos_indices);
    for(size_t h = 0; h < n_layers + 1; h++){
        tensor_free(&workspace->embeddings[h]);
    }
    tensor_free(&workspace->next_token_prob_dist);
    tensor_free(&workspace->output);
}


void model_gpt_init_params(const char *filename, GPTParams *params){
    char *data = read_file(filename);

    char name[128] = "\0";
    snprintf(name, sizeof(name), "wpe.weight");
    params->wpe.weight = safetensors_create_tensor(data,  name);
    tensor_print(&params->wpe.weight, name);

    snprintf(name, sizeof(name), "wte.weight");
    params->wte.weight = safetensors_create_tensor(data,  name);
    tensor_print(&params->wte.weight, name);

    for(size_t h = 0; h < 12; h++){


        // Multi Head Attention
        snprintf(name, sizeof(name), "h.%zu.attn.bias", h);
        params->h[h].attn.bias = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].attn.bias, name);

        snprintf(name, sizeof(name), "h.%zu.attn.c_attn.bias", h);
        params->h[h].attn.c_attn.bias = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].attn.c_attn.bias, name);

        snprintf(name, sizeof(name), "h.%zu.attn.c_attn.weight", h);
        params->h[h].attn.c_attn.weight = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].attn.c_attn.weight, name);

        snprintf(name, sizeof(name), "h.%zu.attn.c_proj.bias", h);
        params->h[h].attn.c_proj.bias = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].attn.c_proj.bias, name);

        snprintf(name, sizeof(name), "h.%zu.attn.c_proj.weight", h);
        params->h[h].attn.c_proj.weight = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].attn.c_proj.weight, name);


        // Multi Layer Perceptron
        snprintf(name, sizeof(name), "h.%zu.mlp.c_fc.bias", h);
        params->h[h].mlp.c_fc.bias = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].mlp.c_fc.bias, name);
        
        snprintf(name, sizeof(name), "h.%zu.mlp.c_fc.weight", h);
        params->h[h].mlp.c_fc.weight = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].mlp.c_fc.weight, name);

        snprintf(name, sizeof(name), "h.%zu.mlp.c_proj.bias", h);
        params->h[h].mlp.c_proj.bias= safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].mlp.c_proj.bias, name);
            
        snprintf(name, sizeof(name), "h.%zu.mlp.c_proj.weight", h);
        params->h[h].mlp.c_proj.weight = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].mlp.c_proj.weight, name);

        //Layer Norms
        snprintf(name, sizeof(name), "h.%zu.ln_%d.bias", h, 1);
        params->h[h].ln_1.bias = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].ln_1.bias, name);

        snprintf(name, sizeof(name), "h.%zu.ln_%d.weight", h, 1);
        params->h[h].ln_1.weight = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].ln_1.weight, name);

        snprintf(name, sizeof(name), "h.%zu.ln_%d.bias", h, 2);
        params->h[h].ln_2.bias= safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].ln_2.bias, name);

        snprintf(name, sizeof(name), "h.%zu.ln_%d.weight", h, 2);
        params->h[h].ln_2.weight = safetensors_create_tensor(data,  name);
        tensor_print(&params->h[h].ln_2.weight, name);
    }


    snprintf(name, sizeof(name), "ln_f.bias");
    params->ln_f.bias = safetensors_create_tensor(data,  name);
    tensor_print(&params->ln_f.bias, name);
    
    snprintf(name, sizeof(name), "ln_f.weight");
    params->ln_f.weight = safetensors_create_tensor(data,  name);
    tensor_print(&params->ln_f.weight, name);

    snprintf(name, sizeof(name), "head.bias");
    params->head.bias  = tensor_init(
        NULL, 
        (uint32_t[]){params->wte.weight.shape[0]}, 
        1, 
        DTYPE_FP32, 
        name
    );
    snprintf(name, sizeof(name), "head.weight");
    params->head.weight  = tensor_transpose(&params->wte.weight);
    strcpy(params->head.weight.name, name);
}

GPTModel model_gpt_init(
    GPTParams *params,
    Vocab *vocab,
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
    strcpy(model.name, name);
    
    model.params = params;
    model.vocab = vocab;
    
    model_gpt_config_init(&model.config, vocab_size, context_len, embed_dim, n_heads, n_layers, drop_rate, qkv_bias, batch_size, dtype, name);
    
    char tmp_name[128] = "\0";
    snprintf(tmp_name, sizeof(tmp_name), "pos_emb");
    model.wpe = embedding_layer_init(&params->wpe, embed_dim, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "token_emb");
    model.wte = embedding_layer_init(&params->wte, embed_dim, tmp_name);

    for(size_t i = 0; i < model.config.n_layers; i++){
        snprintf(tmp_name, sizeof(tmp_name), "h.%zu", i);
        model.h[i] = transformer_layer_init(&params->h[i], n_heads, tmp_name);
    }    
    snprintf(tmp_name, sizeof(tmp_name), "ln_f");
    model.ln_f = layer_norm_init(&params->ln_f, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "head");
    model.head = linear_layer_init(&params->head, tmp_name);

    model_gpt_workspace_init(&model, name);
    return model;
}


void model_gpt_free(GPTModel *model){
    model_gpt_workspace_free(&model->workspace, model->config.n_layers);
    embedding_layer_free(&model->wte);
    embedding_layer_free(&model->wpe);
    for(size_t h = 0; h < model->config.n_layers; h++){
        transformer_layer_free(&model->h[h]);
    }
    layer_norm_free(&model->ln_f);
    linear_layer_free(&model->head);

}

static inline int get_next_token_id(Tensor * next_token_prob_dist){
    float max_prob = -1;
    int token_id = 0;
    for(size_t i = 0; i < next_token_prob_dist->size; i++){
        float prob = ((float*)next_token_prob_dist->data)[i];
        if(prob > max_prob){
            max_prob = prob;
            token_id = i;
        } 
    }
    //printf("max_prob: %.f\n", max_prob);
    return token_id;
}

void model_gpt_forward(GPTModel *model, Tensor *x, const char *prompt){
    assert(x->ndim == 3);

    // for(size_t i = 0; i < vocab.len; i++){
    //     printf("%s\n", vocab.tokens[i].token);
    // }

    // size_t next_token_index = 5;
    size_t max_itrs = 500;
    printf("\n\n%s", prompt);
    fflush(stdout);
    for(size_t itr = 0; itr < max_itrs; itr++){
        //tensor_print(x, "x embeddings");
        embedding_layer_forward(&model->wte, x);

        if(itr == 0){
            tensor_arange_(0, x->shape[x->ndim-1], 1, &model->workspace.pos_indices);
            tensor_unsqueeze_(&model->workspace.pos_indices, 0);
        }
        embedding_layer_forward(&model->wpe, &model->workspace.pos_indices);


        tensor_add_(
            &model->wte.workspace.output, 
            &model->wpe.workspace.output, 
            &model->workspace.embeddings[0]
        );
        //tensor_print(&model->workspace.embeddings[0], " &model->workspace.embeddings[0]");
        for(size_t h = 0; h <  model->config.n_layers; h++){
            transformer_layer_forward(&model->h[h], &model->workspace.embeddings[h]);
            tensor_copy_(&model->h[h].workspace.output, &model->workspace.embeddings[h+1]);
        }

        layer_norm_forward(&model->ln_f, &model->workspace.embeddings[model->config.n_layers]);
        linear_layer_forward(&model->head, &model->ln_f.workspace.output);
        tensor_softmax_(&model->head.workspace.output, 1,  &model->workspace.output);

        // tensor_print(&model->workspace.output, "&model->workspace.output");

        tensor_copy_row_data(
            &model->workspace.next_token_prob_dist, 
            0, 0, 
            &model->workspace.output, 
            model->workspace.output.shape[model->workspace.output.ndim-2]-1, 
            model->config.vocab_size
        );

        int next_token_id = get_next_token_id(&model->workspace.next_token_prob_dist);
        //printf("\n Token | next_token_id: %d, token: %s\n", next_token_id, model->vocab->tokens[next_token_id].token);

        for(size_t i = 1; i < x->size; i++){
            ((int*)x->data)[i-1] = ((int*)x->data)[i];
        }
        ((int*)x->data)[x->size-1] = next_token_id;
        //model_gpt_write(model, "c_model.safetensors");
        printf("%s",  model->vocab->tokens[next_token_id].token);
        fflush(stdout);
    }
    printf("\n");
}


void model_gpt_write(GPTModel *model, const char *filename){
    printf("Saving Model ......\n");
    fflush(stdout);
    Tensor *tensors[2000];
    size_t tensors_len = 0;
    embedding_layer_write(&model->wte, tensors, &tensors_len);
    embedding_layer_write(&model->wpe, tensors, &tensors_len);

    for(size_t h = 0; h < model->config.n_layers; h++){
        tensors[tensors_len++] = &model->workspace.embeddings[h];
        transformer_layer_write(&model->h[h], tensors, &tensors_len);
    }
    tensors[tensors_len++] = &model->workspace.embeddings[model->config.n_layers];

    layer_norm_write(&model->ln_f, tensors, &tensors_len);
    linear_layer_write(&model->head, tensors, &tensors_len);
    
    tensors[tensors_len++] = &model->workspace.output;
    tensors[tensors_len++] = &model->workspace.next_token_prob_dist;
    

    printf("writing %zu tensors to disk\n", tensors_len);

    for(size_t i = 0; i < tensors_len; i++){
        if(tensors[i] != NULL){
            if(tensors[i]->size == 0){
                tensor_print(tensors[i], "Tensor Size is 0");
                exit(1);
            }
        }
        else{
            printf("tensor is NULL with index: %zu", tensors_len);
            exit(1);
        }
    }

    safetensors_save_model(filename, tensors, tensors_len);
}
