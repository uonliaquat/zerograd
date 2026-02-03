#include "../include/safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../include/tensor.h"
#include "../include/utils.h"

static inline void extract_vals(char *start, uint32_t *vals, uint8_t *len){
    uint32_t dim = 0;
    while(1){
        if(start[0] == ',' || start[0] == ']'){
            vals[(*len)++] = dim;
            
            dim = 0;
            if(start[0] == ']') break;
            start++;
        }
        int val = start[0] - '0';
        dim *= 10;
        dim += val;
        start++;
    }
}

static inline Tensor safetensors_create_tensor(char *data, char *t_name){
    uint64_t json_size = 0;
    memcpy(&json_size, data, 8);
    // printf("json_size: %llu\n", json_size);

    char raw_json[json_size];
    memcpy(raw_json, data+8, json_size);
    //printf("%s\n", raw_json);

    // //get name
    char *layer_start = strstr(raw_json, t_name);
    //char *t_name_end = strstr(start, "\"");
    // char t_name[TENSOR_MAX_LEN_NAME] = "\0";
    // memcpy(t_name, start, t_name_end - start);
    // printf("t_name: %s\n", t_name);
    // printf("%.*s\n", 128, start);

    // //Read Shape
    char *shape_start   = strstr(layer_start, "shape") + 8;
    uint32_t shape[TENSOR_MAX_SHAPE_DIM] = {0};
    uint8_t ndim = 0;
    extract_vals(shape_start, shape, &ndim);
    // printf("ndim: %u\n", ndim);
    // printf("shape: (");
    // for(size_t i = 0; i < ndim; i++){
    //     printf("%u, ", shape[i]); 
    // }
    // printf(")\n");


    // // Read data_offsets
    char *data_offsers_start = strstr(shape_start, "data_offsets") + 15;
    uint32_t offset[2] = {0};
    uint8_t len = 0;
    extract_vals(data_offsers_start, offset, &len);
    assert(len == 2);
    // printf("offset: [");
    // for(size_t i = 0; i < len; i++){
    //     printf("%u, ", offset[i]); 
    // }
    // printf("]\n");

    size_t weights_size = (offset[1]-offset[0]);
    //printf("weights_size: %zu\n", weights_size);
    float *weights = calloc(weights_size, 4);
    char *weights_start = data + 8 + json_size + offset[0];
    memcpy(weights, weights_start, weights_size);
    // // for(size_t i = 0; i < 10; i++){
    // //     printf("%f\n", weights[i]);
    // // }
    Tensor my_tensor = tensor_init(weights, shape, ndim, DTYPE_FP32, t_name);
    free(weights);
    return my_tensor;

}

void safetensors_load_model(const char *filename, GPTParameters *gpt_params){
    char *data = read_file(filename);
    char *tmp_data = data;

    char name[64] = "\0";

    snprintf(name, sizeof(name), "wpe.weight");
    gpt_params->wpe.weight = safetensors_create_tensor(tmp_data,  name);
    tensor_print(&gpt_params->wpe.weight, name);

    snprintf(name, sizeof(name), "wte.weight");
    gpt_params->wte.weight = safetensors_create_tensor(tmp_data,  name);
    tensor_print(&gpt_params->wte.weight, name);


    //hidden layers
    for(size_t i = 0; i < 12; i++){

        //attn
        snprintf(name, sizeof(name), "h.%zu.attn.bias", i);
        gpt_params->h[i].attn.bias = safetensors_create_tensor(tmp_data,  name);
        tensor_print(&gpt_params->h[i].attn.bias, name);

        snprintf(name, sizeof(name), "h.%zu.attn.c_attn.bias", i);
        gpt_params->h[i].attn.c_attn.bias = safetensors_create_tensor(tmp_data,  name);
        tensor_print(&gpt_params->h[i].attn.c_attn.bias, name);

        snprintf(name, sizeof(name), "h.%zu.attn.c_attn.weight", i);
        gpt_params->h[i].attn.c_attn.weight = safetensors_create_tensor(tmp_data,  name);
        tensor_print(&gpt_params->h[i].attn.c_attn.weight, name);

        snprintf(name, sizeof(name), "h.%zu.attn.c_proj.bias", i);
        gpt_params->h[i].attn.c_proj.bias = safetensors_create_tensor(tmp_data,  name);
        tensor_print(&gpt_params->h[i].attn.c_proj.bias, name);

        snprintf(name, sizeof(name), "h.%zu.attn.c_proj.weight", i);
        gpt_params->h[i].attn.c_proj.weight = safetensors_create_tensor(tmp_data,  name);
        tensor_print(&gpt_params->h[i].attn.c_proj.weight, name);

        //ln
        for(size_t j = 0; j < 2; j++){
            snprintf(name, sizeof(name), "h.%zu.ln_%zu.bias", i, j+1);
            gpt_params->h[i].ln_[j].bias = safetensors_create_tensor(tmp_data,  name);
            tensor_print(&gpt_params->h[i].ln_[j].bias, name);

            snprintf(name, sizeof(name), "h.%zu.ln_%zu.weight", i, j+1);
            gpt_params->h[i].ln_[j].weight = safetensors_create_tensor(tmp_data,  name);
            tensor_print(&gpt_params->h[i].ln_[j].weight , name);
        }

         //mlp
            //c_fc
            snprintf(name, sizeof(name), "h.%zu.mlp.c_fc.bias", i);
            gpt_params->h[i].mlp.c_fc.bias = safetensors_create_tensor(tmp_data,  name);
            tensor_print(&gpt_params->h[i].mlp.c_fc.bias, name);
            
            snprintf(name, sizeof(name), "h.%zu.mlp.c_fc.weight", i);
            gpt_params->h[i].mlp.c_fc.weight = safetensors_create_tensor(tmp_data,  name);
            tensor_print(&gpt_params->h[i].mlp.c_fc.weight, name);

            //c_proj
            snprintf(name, sizeof(name), "h.%zu.mlp.c_proj.bias", i);
            gpt_params->h[i].mlp.c_proj.bias = safetensors_create_tensor(tmp_data,  name);
            tensor_print(&gpt_params->h[i].mlp.c_proj.bias, name);
             
            snprintf(name, sizeof(name), "h.%zu.mlp.c_proj.weight", i);
            gpt_params->h[i].mlp.c_proj.weight = safetensors_create_tensor(tmp_data,  name);
            tensor_print(&gpt_params->h[i].mlp.c_proj.weight, name);

    }

    //ln_f
    snprintf(name, sizeof(name), "ln_f.bias");
    gpt_params->ln_f.bias = safetensors_create_tensor(tmp_data,  name);
    tensor_print(&gpt_params->ln_f.bias, name);
    
    snprintf(name, sizeof(name), "ln_f.weight");
    gpt_params->ln_f.weight = safetensors_create_tensor(tmp_data,  name);
    tensor_print(&gpt_params->ln_f.weight, name);

}

void safetensors_save_model(const char *filename, GPTParameters *gpt_params){

}