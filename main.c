#include <stdio.h>
#include "./inc/models/gpt2.h"
#include "./inc/graph/graph.h"
#include "./inc/graph/layout.h"

typedef enum DType DType;
int main(){
    
    printf("Running inference engine\n");

    GPT2Config config = {
        .ctx_win = 1024,
        .ndim = 768,
        .vocab_size = 50257,
        .nheads = 12,
        .nlayers = 12,
        .qkv_bias = true,
        .dtype = DTYPE_F32
    };

    
    Graph graph = graph_init(280);
    build_graph_gpt2(&config, &graph);

    size_t nbytes = graph_plan_memory(&graph);
    graph.ctx = context_init(nbytes);
    // graph_load_weights(&graph, "/Users/uonliaquat/workspace/zerograd/gpt2.zg", "/Users/uonliaquat/Downloads/gpt2.safetensors");
    // graph_print(&graph);
    // //set input
    // Tensor *token_ids = &graph.nodes[0];
    // int *token_ids_data = ((int*)((char*)graph.ctx.mem + token_ids->data_offset));
    // // int *token_indices_data = ((int*)((char*)graph.ctx.mem + token_indices->data_offset));
    // for(size_t i = 0; i < token_ids->shape[0]; i++){
    //     token_ids_data[i] = 0;
    //     // token_indices_data[i] = i;
    // }

    
    // graph_execute(&graph);
    // graph_print_weights(&graph);
    // graph_write(&graph, "my_model.safetensors");
    // graph_free(&graph);
    return 0;
}