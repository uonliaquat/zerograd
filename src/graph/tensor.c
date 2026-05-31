#include "../../inc/graph/tensor.h"
#include "../../inc/graph/graph.h"

#include <stdio.h>
Tensor *tensor_create(
    Graph *graph,
    char *name, 
    size_t data_offset, 
    size_t *shape, 
    uint8_t ndim, 
    Tensor **src,
    size_t nsrc,
    DType d_type, 
    OpType op_type
){

    Tensor *out = graph_alloc_node(graph);
    out->id = token_curr_id++;
    strcpy(out->name, name);

    out->data_offset = data_offset;
    out->nbytes = 1;
    for(size_t i = 0; i < ndim; i++){
         out->nbytes *= shape[i];
    }
    out->nbytes *= dtype_size(d_type);
    memset(out->shape, 0, 4);
    memset(out->stride, 1, 4);

    memcpy(out->shape, shape, ndim * sizeof(size_t));
    

    out->nsrc = nsrc;
    for(size_t i = 0; i < 4; i++) out->src[i] = NULL;
    for(size_t i = 0; i < nsrc; i++) out->src[i] = src[i];
    out->d_type = d_type;
    out->op_type = op_type;
    
    return out;
}

void tensor_print_header()
{
    printf("%-4s | %-20s | %-14s | %-42s | %-8s | %-15s\n",
           "ID",
           "NAME",
           "OFFSET",
           "SOURCES",
           "DTYPE",
           "OP");

    printf("-----------------------------------------------------------------------------------------------\n");
}

void tensor_print(const Tensor *t){
    char src_buf[256] = {0};
    for(size_t i = 0; i < t->nsrc; i++){
        strcat(src_buf, t->src[i]->name);
        if(i+1 < t->nsrc) strcat(src_buf, ", ");
    }
    printf("%-4zu | %-20s | 0x%012zu | %-42s | %-8s | %-15s\n",
           t->id,
           t->name,
           t->data_offset,
           src_buf,
           dtype_name(t->d_type),
           op_name(t->op_type)
    );
}