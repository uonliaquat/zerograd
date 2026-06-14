#include "../../inc/ops/op_layernorm.h"
#include "../../inc/kernels/kernel_layernorm.h"
#include "../../inc/graph/op_table.h"
// #include "../../inc/graph/tensor.h"

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>


size_t op_layernorm_scratch_bytes(const Tensor *tensor){
    switch(tensor->d_type){
        case DTYPE_F32: return kernel_layernorm_cpu_f32_scratch_bytes();
        default: return 0;
    }
}

Tensor *op_layernorm(Graph *graph, const char *name,
    Tensor *weight, Tensor *bias, Tensor *input){
        assert(input->ndim == weight->ndim);
        // assert(input->shape[0] == weight->shape[0]);
        size_t ctx_win = input->shape[0];
        size_t ndim = input->shape[1];
        Tensor *out = graph_alloc_node(graph);
        LayerNormParams *op_params = calloc(1, sizeof(LayerNormParams));
        op_params->eps = 1e-5;
        tensor_create(out, name, (size_t[]){ctx_win, ndim}, 2, (Tensor*[]){weight, bias, input}, 3, input->d_type, OP_LAYER_NORM, op_params, TENSOR_ACTIVATION);
        return out;
    }

void op_layernorm_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));

    float *weights  = &ctx->mem[tensor->src[0]->data_offset];
    float *bias     = &ctx->mem[tensor->src[1]->data_offset];
    float *embed    = &ctx->mem[tensor->src[2]->data_offset];
    float *out      = &ctx->mem[tensor->data_offset];

    size_t size_weights     = tensor->src[0]->nelems;
    size_t size_bias        = tensor->src[1]->nelems;
    size_t seq_len          = tensor->src[2]->shape[0];
    size_t embed_dim        = tensor->src[2]->shape[1];

    // printf("size_weights: %zu, size_bias: %zu, seq_len: %zu\n", size_weights, size_bias, seq_len);
    assert(size_weights == size_bias);

    LayerNormParams * params = ((LayerNormParams*)(tensor->op_params));
    kernel_layernorm_cpu_f32_forward(embed, weights, bias, out, seq_len, embed_dim, params->eps);
}