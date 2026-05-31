#ifndef __CONTEXT_H__
#define __CONTEXT_H__

#include <stddef.h>

typedef struct Context {
    void *mem;
    size_t nbytes;
    size_t size;

} Context;


void context_init(size_t nbytes);
// size_t context_alloc(size_t nbytes);

// Tensor *context_get_node(Context *ctx);
#endif
