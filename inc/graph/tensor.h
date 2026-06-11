#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <string.h>
#include <stdint.h>

#include "./op_table.h"



typedef enum DType {
    DTYPE_F32,
    DTYPE_I32
} DType;



typedef struct Tensor {

    // Identitiy
    size_t id;
    char name[128];

    // Shape, type and sizes
    size_t shape[4];
    size_t stride[4];
    uint8_t ndim;
    size_t nelems;
    size_t nbytes;
    DType d_type;


    // Graph Structure
    struct Tensor *src[4];
    size_t nsrc;
    OpType op_type;

    // Memory Location 
    size_t data_offset;
    size_t scratch_offset;
    size_t nbytes_scratch;

    //OP configuration (How to compute)
    void *op_params;

} Tensor;

void tensor_create(
    Tensor *out,
    const char *name, 
    const size_t *shape, 
    const uint8_t ndim, 
    Tensor **src,
    const size_t nsrc,
    const DType d_type,
    const OpType op_type,
    void *op_params
);

void tensor_print_header(void);
void tensor_print(const Context *ctx, const Tensor *t);
// void tensor_print_weights(const Context *ctx, const Tensor *t);

static inline size_t dtype_size(DType d_type){
    switch(d_type){
        case DTYPE_F32: return 4;
        case DTYPE_I32: return 4;
        default: return -1;
    }
}

static inline char *dtype_name(DType d_type){
    switch(d_type){
        case DTYPE_F32: return "F32";
        case DTYPE_I32: return "I32";
        default: return "UNKNOWN"; 
    }
}
#endif
