#include "../../inc/models/gpt2.h"
#include "../../inc/ops/op_input.h"
#include "../../inc/ops/op_weight.h"
#include "../../inc/ops/op_bias.h"
#include "../../inc/ops/op_index.h"
#include "../../inc/ops/op_arange.h"
#include "../../inc/ops/op_add.h"
#include "../../inc/ops/op_linear.h"
#include "../../inc/ops/op_layernorm.h"
#include "../../inc/ops/op_qkvproj.h"
#include "../../inc/ops/op_attention.h"
#include "../../inc/ops/op_gelu.h"


#include <stdio.h>
#include <stdlib.h>




static inline char *layer_name(char *buff, size_t buff_size, size_t i, char *suffix){
    memset(buff, 0, buff_size);
    snprintf(buff, buff_size, "h.%zu.%s", i, suffix);
    return buff;
}



void build_graph_gpt2(GPT2Config *config, Graph *graph){
    char buff[128] = {0};
    DType dtype = config->dtype;
    Tensor *token_ids       = op_input(graph, "token.ids", config->ctx_win);
    Tensor *wte             = op_weight(graph, "wte", config->vocab_size, config->ndim, dtype);
    Tensor *token_indices   = op_arange(graph, "token.indices", config->ctx_win);
    Tensor *wpe             = op_weight(graph, "wpe", config->ctx_win, config->ndim, dtype);
    Tensor *token_embed     = op_index(graph, "token.embed", config->ctx_win, config->ndim, wte, token_ids);
    Tensor *pos_embed       = op_index(graph, "pos.embed", config->ctx_win, config->ndim, wpe, token_indices);
    Tensor *input_embed     = op_add(graph, "input.embed", token_embed, pos_embed);
    
    
    //Transfromer Block
    for(size_t i = 0; i < config->nlayers; i++){
        // printf("========================== LAYER %zu ==========================\n", i);
        Tensor *ln1_weight  = op_weight(graph, layer_name(buff, sizeof(buff), i, "ln1.weight"), 1, config->ndim, dtype);
        Tensor *ln1_bias    = op_bias(graph, layer_name(buff, sizeof(buff), i, "ln1.bias"), config->ndim, dtype);
        Tensor *ln1_out     = op_layernorm(graph, layer_name(buff, sizeof(buff), i, "ln1.out"), ln1_weight, ln1_bias, input_embed);
        

        Tensor *q_weight  = op_weight(graph, layer_name(buff, sizeof(buff), i, "q.weight"), config->ndim, config->ndim, dtype);
        Tensor *q_bias    = op_bias(graph, layer_name(buff, sizeof(buff), i, "q.bias"), config->ndim, dtype);
        Tensor *k_weight  = op_weight(graph, layer_name(buff, sizeof(buff), i, "k.weight"), config->ndim, config->ndim, dtype);
        Tensor *k_bias    = op_bias(graph, layer_name(buff, sizeof(buff), i, "k.bias"), config->ndim, dtype);
        Tensor *v_weight  = op_weight(graph, layer_name(buff, sizeof(buff), i, "v.weight"), config->ndim, config->ndim, dtype);
        Tensor *v_bias    = op_bias(graph, layer_name(buff, sizeof(buff), i, "v.bias"), config->ndim, dtype);

        Tensor *q_proj    = op_linear(graph, layer_name(buff, sizeof(buff), i, "q.proj"), q_weight, q_bias, ln1_out, true);
        Tensor *k_proj    = op_linear(graph, layer_name(buff, sizeof(buff), i, "k.proj"), k_weight, k_bias, ln1_out, true);
        Tensor *v_proj    = op_linear(graph, layer_name(buff, sizeof(buff), i, "v.proj"), v_weight, v_bias, ln1_out, true);

        
        Tensor *attn_out    = op_attention(graph, layer_name(buff, sizeof(buff), i, "attn.out"), config->nheads, q_proj, k_proj, v_proj);
        Tensor *attn_proj_weight = op_weight(graph, layer_name(buff, sizeof(buff), i, "attn.proj.weight"), config->ndim, config->ndim, dtype);
        Tensor *attn_proj_bias = op_bias(graph, layer_name(buff, sizeof(buff), i, "attn.proj.bias"), config->ndim, dtype);
        Tensor *attn_proj = op_linear(graph, layer_name(buff, sizeof(buff), i, "attn.proj"), attn_proj_weight, attn_proj_bias, attn_out, true);

        Tensor *res_conn1 = op_add(graph, layer_name(buff, sizeof(buff), i, "res.conn1"), attn_proj, input_embed);

        Tensor *ln2_weight = op_weight(graph, layer_name(buff, sizeof(buff), i, "ln2.weight"), 1, config->ndim, dtype);
        Tensor *ln2_bias = op_bias(graph, layer_name(buff, sizeof(buff), i, "ln2.bias"), config->ndim, dtype);
        Tensor *ln2_out = op_layernorm(graph, layer_name(buff, sizeof(buff), i, "ln2.out"), ln2_weight, ln2_bias, res_conn1);

        //MLP
        Tensor *mlp_up_weight = op_weight(graph, layer_name(buff, sizeof(buff), i, "mlp.up.weight"), config->ndim, config->ndim*4, dtype);
        Tensor *mlp_up_bias = op_bias(graph, layer_name(buff, sizeof(buff), i, "mlp.up.bias"), config->ndim*4, dtype);
        Tensor *mlp_up_proj = op_linear(graph, layer_name(buff, sizeof(buff), i, "mlp.up.proj"), mlp_up_weight, mlp_up_bias, ln2_out, true);
        Tensor *mlp_up_proj_gelu_out = op_gelu(graph, layer_name(buff, sizeof(buff), i, "mlp.up.proj.gelu.out"), mlp_up_proj);

        Tensor *mlp_down_weight = op_weight(graph, layer_name(buff, sizeof(buff), i, "mlp.down.weight"), config->ndim*4, config->ndim, dtype);
        Tensor *mlp_down_bias = op_bias(graph, layer_name(buff, sizeof(buff), i, "mlp.down.bias"), config->ndim, dtype);
        Tensor *mlp_down_proj = op_linear(graph,layer_name(buff, sizeof(buff), i, "mlp.down.proj"),  mlp_down_weight, mlp_down_bias, mlp_up_proj_gelu_out, true);
        input_embed = op_add(graph, layer_name(buff, sizeof(buff), i, "out"), res_conn1, mlp_down_proj);

    }

    Tensor *ln_weight   = op_weight(graph, "ln.weight", 1, config->ndim, dtype);
    Tensor *ln_bias     = op_bias(graph, "ln.bias", config->ndim, dtype);
    Tensor *ln_out      = op_layernorm(graph, "ln.out", ln_weight, ln_bias, input_embed);

    Tensor *lm_head     = op_linear(graph, "lm.head", wte, NULL, ln_out, false);

}




