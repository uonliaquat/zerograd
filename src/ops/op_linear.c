#include "../../inc/ops/op_linear.h"
#include "../../inc/kernels/kernel_linear.h"
#include "../../inc/graph/op_table.h"
// #include "../../inc/graph/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>


size_t op_linear_scratch_bytes(const Tensor *tensor){
    switch(tensor->d_type){
        case DTYPE_F32: return kernel_linear_cpu_f32_scratch_bytes();
        default: return 0;
    }
}

Tensor *op_linear(
    Graph *graph, const char *name,
    Tensor *weight, Tensor *bias, Tensor *input,
    bool trans_weight
){
    if(trans_weight) assert(input->shape[1] == weight->shape[0]);
    else assert(input->shape[0] == weight->shape[0]);
    size_t rows = input->shape[0];
    size_t cols = weight->shape[1];
    Tensor *out = graph_alloc_node(graph);

    LinearParams *op_params = calloc(1, sizeof(LinearParams));
    op_params->trans_weight = trans_weight;
    op_params->is_bias = bias != NULL ? true : false;

    if(bias != NULL)
        tensor_create(out, name, (size_t[]){rows, cols}, 2, (Tensor*[]){weight, bias, input}, 3, input->d_type, OP_LINEAR, op_params);
    else
        tensor_create(out, name, (size_t[]){rows, cols}, 2, (Tensor*[]){weight, input}, 2, input->d_type, OP_LINEAR, op_params);
    return out;
}

void op_linear_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));

    assert(tensor->op_params != NULL);

    float *weight   = &ctx->mem[tensor->src[0]->data_offset];
    float *bias     = &ctx->mem[tensor->src[1]->data_offset];
    float *input    = &ctx->mem[tensor->src[2]->data_offset];
    float *out      = &ctx->mem[tensor->data_offset];

    size_t rows_weight  = tensor->src[0]->shape[0];
    size_t cols_weight  = tensor->src[0]->shape[1];
    size_t size_bias    = tensor->src[1]->nelems;
    size_t rows_input   = tensor->src[2]->shape[0];
    size_t cols_input   = tensor->src[2]->shape[1];
    size_t rows_out     = tensor->shape[0];
    size_t cols_out     = tensor->shape[1];

    // printf("cols_input: %zu, rows_weight: %zu, rows_input: %zu, rows_out: %zu, cols_weight: %zu, cols_out: %zu, size_bias: %zu\n\n", 
    //         cols_input,      rows_weight,      rows_input,      rows_out,      cols_weight,      cols_out,      size_bias);
    assert(cols_input == rows_weight && rows_input == rows_out && cols_weight == cols_out && size_bias == cols_out);
    kernel_linear_cpu_f32_forward(
        weight, 
        bias,
        input,
        out, 
        rows_out, cols_out, cols_input, 
        ((LinearParams*)(tensor->op_params))->trans_weight
    );

}