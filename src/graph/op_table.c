#include "../../inc/graph/op_table.h"

OpVTable OpTable[7] = {
    [OP_NONE]       = {.forward = NULL, .scratch_bytes = NULL},
    [OP_INDEX]      = {.forward = &op_index_forward,        .scratch_bytes = &op_index_scratch_bytes},
    [OP_ADD]        = {.forward = &op_add_forward,          .scratch_bytes = &op_add_scratch_bytes},
    [OP_LAYER_NORM] = {.forward = &op_layernorm_forward,    .scratch_bytes = &op_layernorm_scratch_bytes},
    [OP_LINEAR]     = {.forward = &op_linear_forward,        .scratch_bytes = &op_linear_scratch_bytes},
    [OP_ATTENTION]  = {.forward = &op_attention_forward,    .scratch_bytes = &op_attention_scratch_bytes},
    [OP_GELU]       = {.forward = &op_gelu_forward,         .scratch_bytes = &op_gelu_scratch_bytes}
};