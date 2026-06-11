#include "../../inc/ops/op_qkvproj.h"
#include "../../inc/kernels/kernel_qkvproj.h"
#include "../../inc/graph/op_table.h"


size_t op_qkv_proj_scratch_bytes(const Tensor *tensor){
    switch(tensor->d_type){
        case DTYPE_F32: return kernel_qkv_proj_cpu_f32_scratch_bytes();
        default: return 0;
    }
}

Tensor *op_qkv_proj(
    Graph *graph, const char *name,
    Tensor *weight, Tensor *bias, Tensor *input
){

    size_t rows = input->shape[0];
    size_t cols = weight->shape[1];
    Tensor *out = graph_alloc_node(graph);

    // LinearParams *op_params = calloc(1, sizeof(LinearParams));
    // op_params->trans_weight = trans_weight;
    // op_params->is_bias = bias != NULL ? true : false;

    if(bias != NULL)
        tensor_create(out, name, (size_t[]){rows, cols}, 2, (Tensor*[]){weight, bias, input}, 3, input->d_type, OP_QKV_PROJ, NULL);
    else
        tensor_create(out, name, (size_t[]){rows, cols}, 2, (Tensor*[]){weight, input}, 2, input->d_type, OP_QKV_PROJ, NULL);
    return out;
}

void op_qkv_proj_forward(Context *ctx, Tensor *tensor){

}