#include "../../inc/ops/op_gelu.h"
#include "../../inc/graph/op_table.h"
#include "../../inc/graph/tensor.h"
#include <stdio.h>

void op_gelu_forward(Context *ctx, Tensor *tensor){
    printf("Executing %s\n", op_name(tensor->op_type));
}