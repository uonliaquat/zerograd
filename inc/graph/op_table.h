#ifndef __OP_TABLE_H__
#define __OP_TABLE_H__

#include <string.h>
#include <stdbool.h>


typedef struct Tensor Tensor;
typedef struct Context Context;

typedef enum OpType {
    OP_NONE,
    OP_INDEX,
    OP_ARANGE,
    OP_ADD,
    OP_LAYER_NORM,
    OP_LINEAR,
    OP_ATTENTION,
    OP_GELU
} OpType;


// typedef struct IndexParams {

// } IndexParams;

// typedef struct ArangePrams {

// } ArangePrams;

// typedef struct AddParams {

// } AddParams;

// typedef struct LayerNormParams {

// } LayerNormParams;

typedef struct LinearParams {
    bool trans_weight;
    bool is_bias;
} LinearParams;

typedef struct AttentionParams{
    size_t embed_dim;
    size_t head_dim;
    size_t n_heads;
    size_t ctx_win;
} AttentionParams;

// typedef struct GeluParams {

// } GeluParams;

// typedef struct InputPrams {

// } InputPrams;

// typedef struct WeightParams {

// } WeightParams;




// extern OpParams op_params[8];


typedef struct OpVTable{
    void (*forward)(Context *, Tensor*);
    size_t (*scratch_bytes)(const Tensor *);
} OpVTable;

extern OpVTable OpTable[8];

static inline char *op_name(OpType op_type){
    switch(op_type){
        case OP_NONE:       return "OP_NONE";
        case OP_INDEX:      return "OP_INDEX";
        case OP_ARANGE:     return "OP_ARANGE";
        case OP_ADD:        return "OP_ADD";
        case OP_LAYER_NORM: return "OP_LAYER_NORM";
        case OP_LINEAR:     return "OP_LINEAR";
        case OP_ATTENTION:  return "OP_ATTENTION";
        case OP_GELU:       return "OP_GELU";
        default:            return "UNKNOWN";
    }
}





#endif
