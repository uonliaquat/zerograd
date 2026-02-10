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

Tensor safetensors_create_tensor(char *data, char *t_name){
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

// void safetensors_load_model(const char *filename, GPTParams *params){
//     char *data = read_file(filename);
//     char *tmp_data = data;

//     char name[64] = "\0";

//     snprintf(name, sizeof(name), "wpe.weight");
//     params->wpe.weight = safetensors_create_tensor(tmp_data,  name);
//     //tensor_print(&params->wpe.weight);

//     snprintf(name, sizeof(name), "wte.weight");
//     params->wte.weight = safetensors_create_tensor(tmp_data,  name);
//     //tensor_print(&params->wte.weight);
//     //hidden layers
//     for(size_t i = 0; i < 12; i++){
//         //attn
//         snprintf(name, sizeof(name), "h.%zu.attn.bias", i);
//         params->h[i].attn.bias = safetensors_create_tensor(tmp_data,  name);

//         snprintf(name, sizeof(name), "h.%zu.attn.c_attn.bias", i);
//         params->h[i].attn.c_attn.bias = safetensors_create_tensor(tmp_data,  name);
//         //tensor_print(&params->h[i].attn.c_attn.bias);

//         snprintf(name, sizeof(name), "h.%zu.attn.c_attn.weight", i);
//         params->h[i].attn.c_attn.weight = safetensors_create_tensor(tmp_data,  name);
//         //tensor_print(&params->h[i].attn.c_attn.weight);

//         snprintf(name, sizeof(name), "h.%zu.attn.c_proj.bias", i);
//         params->h[i].attn.c_proj.bias = safetensors_create_tensor(tmp_data,  name);
//         //tensor_print(&params->h[i].attn.c_proj.bias);

//         snprintf(name, sizeof(name), "h.%zu.attn.c_proj.weight", i);
//         params->h[i].attn.c_proj.weight = safetensors_create_tensor(tmp_data,  name);
//         //tensor_print(&params->h[i].attn.c_proj.weight);

//         //ln
//             snprintf(name, sizeof(name), "h.%zu.ln_%d.bias", i, 1);
//             params->h[i].ln_1.bias = safetensors_create_tensor(tmp_data,  name);
//             //tensor_print(&params->h[i].ln_[j].bias);

//             snprintf(name, sizeof(name), "h.%zu.ln_%d.weight", i, 1);
//             params->h[i].ln_1.weight = safetensors_create_tensor(tmp_data,  name);

//             snprintf(name, sizeof(name), "h.%zu.ln_%d.bias", i, 2);
//             params->h[i].ln_2.bias = safetensors_create_tensor(tmp_data,  name);
//             //tensor_print(&params->h[i].ln_[j].bias);

//             snprintf(name, sizeof(name), "h.%zu.ln_%d.weight", i, 2);
//             params->h[i].ln_2.weight = safetensors_create_tensor(tmp_data,  name);
            
//          //mlp
//             //c_fc
//             snprintf(name, sizeof(name), "h.%zu.mlp.c_fc.bias", i);
//             params->h[i].mlp.c_fc.bias = safetensors_create_tensor(tmp_data,  name);
//             //tensor_print(&params->h[i].mlp.c_fc.bias);
            
//             snprintf(name, sizeof(name), "h.%zu.mlp.c_fc.weight", i);
//             params->h[i].mlp.c_fc.weight = safetensors_create_tensor(tmp_data,  name);
//             //tensor_print(&params->h[i].mlp.c_fc.weight);

//             //c_proj
//             snprintf(name, sizeof(name), "h.%zu.mlp.c_proj.bias", i);
//             params->h[i].mlp.c_proj.bias = safetensors_create_tensor(tmp_data,  name);
//             //tensor_print(&params->h[i].mlp.c_proj.bias);
             
//             snprintf(name, sizeof(name), "h.%zu.mlp.c_proj.weight", i);
//             params->h[i].mlp.c_proj.weight = safetensors_create_tensor(tmp_data,  name);
//             //tensor_print(&params->h[i].mlp.c_proj.weight);

//     }

//     //ln_f
//     snprintf(name, sizeof(name), "ln_f.bias");
//     params->ln_f.bias = safetensors_create_tensor(tmp_data,  name);
//     //tensor_print(&params->ln_f.bias, name);
    
//     snprintf(name, sizeof(name), "ln_f.weight");
//     params->ln_f.weight = safetensors_create_tensor(tmp_data,  name);
//     //tensor_print(&params->ln_f.weight, name);

//     params->head.weight = tensor_transpose(&params->wte.weight);

//     params->head.bias   = tensor_init(NULL, (uint32_t[]){params->wte.weight.shape[0]}, 1, DTYPE_FP32, "head.bias");

// }

static inline size_t get_json_len(Tensor *t, char *json, uint32_t offset_start, uint32_t offset_end) {
    uint64_t json_len;
    uint32_t t_size = t->size * (t->elem_size); 

    size_t pos = 0;
    size_t json_size = 20000;

    /* JSON header */
    pos += snprintf(
        json + pos, json_size - pos,
        "{\"%s\":{\"dtype\":\"%s\",\"data_offsets\":[%u,%u],\"shape\":[",
        t->name,
        tensor_dtype_name(t->dtype),
        offset_start,
        offset_end
    );

    /* shape array */
    for (uint8_t i = 0; i < t->ndim; i++) {
        pos += snprintf(
            json + pos,
            json_size - pos,
            "%s%u",
            (i == 0) ? "" : ",",
            t->shape[i]
        );
    }

    /* close JSON */
    pos += snprintf(
        json + pos, json_size - pos,
        "]}}"
    );

    /* --- DEBUG PRINTS --- */
    printf("Tensor name: %s\n", t->name);
    printf("Tensor dtype: %s\n", tensor_dtype_name(t->dtype));
    printf("Data offsets: [%u, %u]\n", offset_start, offset_end);
    printf("Shape: [");
    for (uint8_t i = 0; i < t->ndim; i++) {
        printf("%s%u", (i == 0) ? "" : ",", t->shape[i]);
    }
    printf("]\n");
    printf("Generated JSON: %s\n", json);

    json_len = pos;
    return json_len;
}

void safetensors_save_model(const char *filename, Tensor **tensors, size_t no_of_tensors){ 
    FILE *fptr = fopen(filename, "w"); // fresh file
    fclose(fptr);

    fptr = fopen(filename, "a");
    if(!fptr){
        perror("Error opening file");
        exit(-1);
    }
    uint64_t json_len;


    size_t pos = 0;
    uint32_t curr_offset = 0;
    uint32_t prev_offset = 0;
    char json[1000000] = "\0";
    size_t json_size = sizeof(json);

    //size_t max_tensors_to_save = 2;
    for(size_t i = 0; i < no_of_tensors; i++){
        Tensor * t = tensors[i];
        uint32_t t_size = t->size * (t->elem_size);
        curr_offset += t_size;
        /* JSON header */
        if(i == 0){
            pos += snprintf(
                json + pos, json_size - pos,
                "{"
            );
        }
        pos += snprintf(
            json + pos, json_size - pos,
            "\"%s\":{\"dtype\":\"%s\",\"data_offsets\":[%u,%u],\"shape\":[",
            t->name,
            tensor_dtype_name(t->dtype),
            prev_offset,
            curr_offset
        );

        /* shape array */
        for (uint8_t i = 0; i < t->ndim; i++) {
            pos += snprintf(
                json + pos,
                json_size - pos,
                "%s%u",
                (i == 0) ? "" : ",",
                t->shape[i]
            );
        }

        /* close JSON */
        pos += snprintf(
            json + pos, json_size - pos,
            (i == no_of_tensors - 1) ? "]}":"]},"
        );

        prev_offset = curr_offset; // probbaly needs to add 1
    }

    pos += snprintf(
        json + pos, json_size - pos,
        "}"
    );

    json_len = pos;

    printf("json_len %llu\n", json_len);

    //strcat(wpe_json, wte_json);
    fwrite(&json_len, 8, 1, fptr);         // header
    fwrite(json, 1, json_len, fptr);       // json
    for(size_t i = 0; i < no_of_tensors; i++){
        Tensor * t = tensors[i];
        uint32_t t_size = t->size * (t->elem_size);
        fwrite(t->data, 1, t_size, fptr);
    }
    fclose(fptr);

}
