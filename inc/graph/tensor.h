#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <string.h>
#include <stdint.h>

#include "./op_table.h"

typedef struct Graph Graph;

static size_t token_curr_id = 0;

typedef enum DType {
    DType_F32,
    DType_I32
} DType;



typedef struct Tensor {
    size_t id;
    char name[128];

    size_t data_offset;
    size_t nbytes;
    size_t nelems;

    size_t shape[4];
    size_t stride[4];
    uint8_t ndim;

    struct Tensor *src[4];
    size_t nsrc;
    DType d_type;
    OpType op_type;

} Tensor;

Tensor *tensor_create(
    Graph *graph,
    char *name, 
    size_t *shape, 
    uint8_t ndim, 
    Tensor **src,
    size_t nsrc,
    DType d_type,
    OpType op_type
);

void tensor_print_header();
void tensor_print(const Tensor *t);
void tensor_print_weights(const Context *ctx, const Tensor *t);

static inline size_t dtype_size(DType d_type){
    switch(d_type){
        case DType_F32: return 4;
        case DType_I32: return 4;
        default: return -1;
    }
}

static inline char *dtype_name(DType d_type){
    switch(d_type){
        case DType_F32: return "F32";
        case DType_I32: return "I32";
        default: return "UNKNOWN"; 
    }
}
#endif
