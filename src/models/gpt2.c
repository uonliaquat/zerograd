#include "../../inc/models/gpt2.h"
#include "../../inc/graph/graph.h"



void init_gpt2_offsets(){

}

void build_gpt2(GPT2Config *config){
    Graph graph = graph_init(NULL, 20);

    size_t data_offset = 0;
    Tensor *token_ids = tensor_create(&graph, "token.ids", data_offset, (size_t[]){config->ctx_win}, 1, NULL, 0, DType_I32, OP_NONE);

    data_offset += (dtype_size(DType_F32) * config->ctx_win);
    Tensor *wte = tensor_create(&graph, "wte", data_offset, (size_t[]){config->vocab_size, config->ndim}, 2, NULL, 0, DType_F32, OP_NONE);

    data_offset += (dtype_size(DType_F32) * config->vocab_size * config->ndim);
    Tensor *wpe = tensor_create(&graph, "wpe", data_offset, (size_t[]){config->ctx_win, config->ndim}, 2, NULL, 0, DType_F32, OP_NONE);

    data_offset += (dtype_size(DType_F32) * config->ctx_win * config->ndim);
    Tensor *token_embed = tensor_create(&graph, "token.embed", data_offset, (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){wte}, 1, DType_F32, OP_INDEX);

    data_offset += (dtype_size(DType_F32) * config->ctx_win * config->ndim);
    Tensor *pos_embed = tensor_create(&graph, "pos.embed", data_offset, (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){wpe}, 1, DType_F32, OP_INDEX);

    data_offset += (dtype_size(DType_F32) * config->ctx_win * config->ndim);
    Tensor *input_embed = tensor_create(&graph, "input.embed", data_offset, (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){wte, wpe}, 2, DType_F32, OP_ADD);
    
    //Transfromer Block
    data_offset += (dtype_size(DType_F32) * config->ctx_win * config->ndim);
    Tensor *ln1_weight = tensor_create(&graph, "h.0.ln1.weight", data_offset, (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);

    data_offset += (dtype_size(DType_F32) * config->ndim);
    Tensor *ln1_bias = tensor_create(&graph, "h.0.ln1.bias", data_offset, (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);
    
    data_offset += (dtype_size(DType_F32) * config->ndim);
    Tensor *ln1_output = tensor_create(&graph, "h.0.ln1.output", data_offset, (size_t[]){config->ndim}, 1, (Tensor*[]){ln1_weight, ln1_bias, input_embed}, 3, DType_F32, OP_LAYER_NORM);
    
    data_offset += (dtype_size(DType_F32) * config->ndim);
    Tensor *attn_weight = tensor_create(&graph, "h.0.attn.weight", data_offset, (size_t[]){config->ndim, config->ndim*3}, 2, NULL, 0, DType_F32, OP_NONE);
    
    data_offset += (dtype_size(DType_F32) * config->ndim *  config->ndim*3);
    Tensor *attn_bias = tensor_create(&graph, "h.0.attn.bias", data_offset, (size_t[]){config->ndim*3}, 1, NULL, 0, DType_F32, OP_NONE);

    data_offset += (dtype_size(DType_F32) * config->ndim * 3);
    Tensor *qkv_proj = tensor_create(&graph, "h.0.qkv_proj", data_offset, (size_t[]){config->ctx_win, config->ndim*3}, 2, NULL, 0, DType_F32, OP_LINEAR);







    graph_print(&graph);
}





