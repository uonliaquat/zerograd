#include <stdio.h>
#include "./inc/models/gpt2.h"
int main(){
    printf("Running inference engine\n");
    GPT2Config config = {
        .ctx_win = 10,
        .ndim = 768,
        .vocab_size = 50257,
        .nheads = 12,
        .nlayers = 12,
        .qkv_bias = true
    };
    build_graph_gpt2(&config);
    return 0;
}