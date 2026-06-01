#include "../../inc/graph/tensor.h"
#include "../../inc/graph/graph.h"

#include <stdio.h>
Tensor *tensor_create(
    Graph *graph,
    char *name, 
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

    out->data_offset = 0;
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

void tensor_print_header(void)
{
    printf("%-4s %-32s %-8s %-15s %-12s\n",
           "ID",
           "NAME",
           "DTYPE",
           "OP",
           "OFFSET");

    printf("---------------------------------------------------------------------------------------\n");
}

void tensor_print(const Tensor *t)
{
    printf("%-4zu %-32s %-8s %-15s %-12zu\n",
           t->id,
           t->name,
           dtype_name(t->d_type),
           op_name(t->op_type),
           t->data_offset);

    for (size_t i = 0; i < t->nsrc; i++) {
        printf("      %s %s\n",
               (i + 1 == t->nsrc) ? "└──" : "├──",
               t->src[i]->name);
    }
}