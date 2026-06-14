#include <stdio.h>
#include <stdlib.h>
#include "./inc/models/gpt2.h"
#include "./inc/graph/graph.h"
#include "./inc/graph/layout.h"

typedef enum DType DType;
int main(int argc, char *argv[]){
    if(argc < 2){
        printf("Uasege: %s \"prompt\"\n", argv[0]);
        return 1;
    }

    GPT2Config config = { 
        .ctx_win = 1024,
        .ndim = 768,
        .vocab_size = 50257,
        .nheads = 12,
        .nlayers = 12,
        .qkv_bias = true,
        .dtype = DTYPE_F32
    };

    
    Graph graph = graph_init(346);
    build_graph_gpt2(&config, &graph);

    size_t nbytes = graph_plan_memory(&graph);
    printf("nbytes=%zu\n", nbytes);
    graph.ctx = context_init(nbytes);
    graph_load_weights(&graph, "/Users/uonliaquat/workspace/zerograd/gpt2_split.safetensors");
    // //set input
    Tensor *token_ids = &graph.nodes[0];
    int *token_ids_data = ((int*)((char*)graph.ctx.mem + token_ids->data_offset));
    // // int *token_indices_data = ((int*)((char*)graph.ctx.mem + token_indices->data_offset));
    for(size_t i = 0; i < token_ids->shape[0]; i++){
        token_ids_data[i] = 0;
        // token_indices_data[i] = i;
    }

    for(size_t i = 1; i < argc; i++){
        token_ids_data[i-1] = atoi(argv[i]);
        // token_indices_data[i] = i;
    }

    
    graph_execute(&graph, argc-1);
    //graph_print(&graph);
    // graph_write(&graph, "my_model.safetensors");
    // graph_free(&graph);
    return 0;
}