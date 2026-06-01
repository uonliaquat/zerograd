#include "../../inc/models/gpt2.h"
#include "../../inc/graph/graph.h"

#include <stdio.h>

static inline char *layer_name(char *buff, size_t buff_size, size_t i, char *suffix){
    memset(buff, 0, buff_size);
    snprintf(buff, buff_size, "h.%zu.%s", i, suffix);

    return buff;
}

void init_gpt2_offsets(){

}

void build_gpt2(GPT2Config *config){
    char buff[128] = {0};
    Graph graph = graph_init(NULL, 300);

    Tensor *token_ids = tensor_create(&graph, "token.ids", (size_t[]){config->ctx_win}, 1, NULL, 0, DType_I32, OP_NONE);
    Tensor *wte = tensor_create(&graph, "wte", (size_t[]){config->vocab_size, config->ndim}, 2, NULL, 0, DType_F32, OP_NONE);
    Tensor *token_indices = tensor_create(&graph, "token.indices", (size_t[]){config->ctx_win}, 1, NULL, 0, DType_I32, OP_NONE);
    Tensor *wpe = tensor_create(&graph, "wpe", (size_t[]){config->ctx_win, config->ndim}, 2, NULL, 0, DType_F32, OP_NONE);
    Tensor *token_embed = tensor_create(&graph, "token.embed", (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){wte, token_ids}, 2, DType_F32, OP_INDEX);
    Tensor *pos_embed = tensor_create(&graph, "pos.embed", (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){wpe, token_indices}, 2, DType_F32, OP_INDEX);
    Tensor *input_embed = tensor_create(&graph, "input.embed", (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){token_embed, pos_embed}, 2, DType_F32, OP_ADD);
    
    //Transfromer Block
    for(size_t i = 0; i < config->nlayers; i++){
        Tensor *ln1_weight = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "ln1.weight"), (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);
        Tensor *ln1_bias = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "ln1.bias"), (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);
        Tensor *ln1_out = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "ln1.out"), (size_t[]){config->ctx_win ,config->ndim}, 2, (Tensor*[]){ln1_weight, ln1_bias, input_embed}, 3, DType_F32, OP_LAYER_NORM);
        
        Tensor *qkv_weight = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "qkv.weight"), (size_t[]){config->ndim, config->ndim*3}, 2, NULL, 0, DType_F32, OP_NONE);
        Tensor *qkv_bias = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "qkv.bias"), (size_t[]){config->ndim*3}, 1, NULL, 0, DType_F32, OP_NONE);
        Tensor *qkv_proj = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "qkv.proj"), (size_t[]){config->ctx_win, config->ndim*3}, 2, (Tensor*[]){qkv_weight, qkv_bias, ln1_out}, 3, DType_F32, OP_LINEAR);
        
        Tensor *attn_out = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "attn.out"), (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){qkv_proj}, 1, DType_F32, OP_ATTENTION);
        
        Tensor *attn_proj_weight = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "attn.proj.weight"), (size_t[]){config->ndim, config->ndim}, 2, NULL, 0, DType_F32, OP_NONE);
        Tensor *attn_proj_bias = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "attn.proj.bias"), (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);
        Tensor *attn_proj = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "attn.proj"), (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){attn_proj_weight, attn_proj_bias, attn_out}, 3, DType_F32, OP_LINEAR);

        Tensor *res_conn1 = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "res.conn1"), (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){attn_proj, input_embed}, 2, DType_F32, OP_ADD);

        Tensor *ln2_weight = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "ln2.weight"), (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);
        Tensor *ln2_bias = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "ln2.bias"), (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);
        Tensor *ln2_out = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "ln2.out"), (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){ln2_weight, ln2_bias, res_conn1}, 3, DType_F32, OP_LAYER_NORM);

        //MLP
        Tensor *mlp_up_weight = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "mlp.up.weight"), (size_t[]){config->ndim, config->ndim*4}, 2, NULL, 0, DType_F32, OP_NONE);
        Tensor *mlp_up_bias = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "mlp.up.bias"), (size_t[]){config->ndim*4}, 1, NULL, 0, DType_F32, OP_NONE);
        Tensor *mlp_up_proj = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "mlp.up.proj"), (size_t[]){config->ctx_win, config->ndim*4}, 2, (Tensor*[]){mlp_up_weight, mlp_up_bias, ln2_out}, 3, DType_F32, OP_LINEAR);
        Tensor *mlp_up_proj_gelu_out = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "mlp.up.proj.gelu.out"), (size_t[]){config->ctx_win, config->ndim*4}, 2, (Tensor*[]){mlp_up_proj}, 1, DType_F32, OP_GELU);
        Tensor *mlp_down_weight = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "mlp.down.weight"), (size_t[]){config->ndim*4, config->ndim}, 2, NULL, 0, DType_F32, OP_NONE);
        Tensor *mlp_down_bias = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "mlp.down.bias"), (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);
        Tensor *mlp_down_proj = tensor_create(&graph,layer_name(buff, sizeof(buff), i, "mlp.down.proj"), (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){mlp_down_weight, mlp_down_bias, mlp_up_proj_gelu_out}, 3, DType_F32, OP_LINEAR);
        input_embed = tensor_create(&graph, layer_name(buff, sizeof(buff), i, "out"), (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){res_conn1, mlp_down_proj}, 2, DType_F32, OP_ADD);

    }

    Tensor *ln_weight   = tensor_create(&graph, "ln.weight", (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);
    Tensor *ln_bias     = tensor_create(&graph, "ln.bias", (size_t[]){config->ndim}, 1, NULL, 0, DType_F32, OP_NONE);
    Tensor *ln_out      = tensor_create(&graph, "ln.out", (size_t[]){config->ctx_win, config->ndim}, 2, (Tensor*[]){ln_weight, ln_bias, input_embed}, 3, DType_F32, OP_LAYER_NORM);

    Tensor *lm_head     = tensor_create(&graph, "lm.head", (size_t[]){config->ctx_win, config->vocab_size}, 2, (Tensor*[]){wte, ln_out}, 2, DType_F32, OP_LINEAR);


    graph_plan_memory(&graph);
    graph_print(&graph);
    //graph_export_dot(&graph, "graph.dot");
    //graph_export_mermaid(&graph, "graph.md");
    // graph_export_json(&graph, "graph.json");
}





