#ifndef __OP_TABLE_H__
#define __OP_TABLE_H__


#include "./../ops/op_index.h"
#include "./../ops/op_add.h"
#include "./../ops/op_layernorm.h"
#include "./../ops/op_linear.h"
#include "./../ops/op_attention.h"
#include "./../ops/op_gelu.h"

typedef struct Tensor Tensor;
typedef struct Context Context;

typedef enum OpType {
    OP_NONE,
    OP_INDEX,
    OP_ADD,
    OP_LAYER_NORM,
    OP_LINEAR,
    OP_ATTENTION,
    OP_GELU
} OpType;


typedef struct OpVTable{
    void (*forward)(Context *, Tensor*);
    size_t (*scratch_bytes)();
} OpVTable;


static inline char *op_name(OpType op_type){
    switch(op_type){
        case OP_NONE:       return "OP_NONE";
        case OP_INDEX:      return "OP_INDEX";
        case OP_ADD:        return "OP_ADD";
        case OP_LAYER_NORM: return "OP_LAYER_NORM";
        case OP_LINEAR:     return "OP_LINEAR";
        case OP_ATTENTION:  return "OP_ATTENTION";
        case OP_GELU:       return "OP_GELU";
        default:            return "UNKNOWN";
    }
}

extern OpVTable OpTable[7];



#endif
