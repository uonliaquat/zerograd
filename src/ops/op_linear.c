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
    else assert(input->shape[1] == weight->shape[1]);
    size_t input_rows = input->shape[0];
    size_t weight_rows = weight->shape[0];
    size_t weight_cols = weight->shape[1];
    Tensor *out = graph_alloc_node(graph);

    LinearParams *op_params = calloc(1, sizeof(LinearParams));
    op_params->trans_weight = trans_weight;
    op_params->is_bias = bias != NULL ? true : false;

    size_t shape[2];
    if(trans_weight){
        shape[0] = input_rows;
        shape[1] = weight_cols;
    }
    else{
        shape[0] = input_rows;
        shape[1] = weight_rows;
    }
  
    tensor_create(out, name, shape, 2, (Tensor*[]){weight, bias, input}, 3, input->d_type, OP_LINEAR, op_params, TENSOR_ACTIVATION);
    return out;
}

void op_linear_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));

    LinearParams *params = ((LinearParams*)(tensor->op_params));
    assert(params != NULL);



    float *weight   = &ctx->mem[tensor->src[0]->data_offset];
    float *bias = NULL;
    if(params->is_bias)
        bias     = &ctx->mem[tensor->src[1]->data_offset];
    float *input    = &ctx->mem[tensor->src[2]->data_offset];
    float *out      = &ctx->mem[tensor->data_offset];

    // size_t rows_weight  = tensor->src[0]->shape[0];
    // size_t cols_weight  = tensor->src[0]->shape[1];
    // size_t size_bias    = tensor->src[1]->nelems;
    // size_t rows_input   = tensor->src[2]->shape[0];

    size_t cols_input   = tensor->src[2]->shape[1];
    size_t rows_out     = tensor->shape[0];
    size_t cols_out     = tensor->shape[1];

    kernel_linear_cpu_f32_forward(
        input,
        weight, 
        bias,
        out, 
        rows_out, cols_out, cols_input, 
        params->trans_weight
    );

}