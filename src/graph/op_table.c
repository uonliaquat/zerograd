#include "../../inc/graph/op_table.h"
#include "../../inc/ops/op_index.h"
#include "../../inc/ops/op_add.h"
#include "../../inc/ops/op_arange.h"
#include "../../inc/ops/op_layernorm.h"
#include "../../inc/ops/op_linear.h"
#include "../../inc/ops/op_attention.h"
#include "../../inc/ops/op_gelu.h"

OpVTable OpTable[8] = {
    [OP_NONE]       = {.forward = NULL, .scratch_bytes = NULL},
    [OP_INDEX]      = {.forward = &op_index_forward,        .scratch_bytes = &op_index_scratch_bytes},
    [OP_ARANGE]     = {.forward = &op_arange_forward,       .scratch_bytes = &op_arange_scratch_bytes},
    [OP_ADD]        = {.forward = &op_add_forward,          .scratch_bytes = &op_add_scratch_bytes},
    [OP_LAYER_NORM] = {.forward = &op_layernorm_forward,    .scratch_bytes = &op_layernorm_scratch_bytes},
    [OP_LINEAR]     = {.forward = &op_linear_forward,        .scratch_bytes = &op_linear_scratch_bytes},
    [OP_ATTENTION]  = {.forward = &op_attention_forward,    .scratch_bytes = &op_attention_scratch_bytes},
    [OP_GELU]       = {.forward = &op_gelu_forward,         .scratch_bytes = &op_gelu_scratch_bytes}
};


// OpParams op_params[8] = {
//     [OP_NONE] = {NULL},
//     [OP_INDEX] = {.index_params-},

// }