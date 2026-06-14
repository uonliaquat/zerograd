#ifndef __LOADER_SAFETENSORS_H__
#define __LOADER_SAFETENSORS_H__

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int       fd;
    size_t    file_size;
    uint8_t  *base;        /* mmap of whole file            */
    char     *json;        /* -> header JSON (base + 8)     */
    size_t    json_len;
    uint8_t  *data;        /* -> tensor block (base+8+N)    */
} SafeTensors;

int st_open(SafeTensors *st, const char *path);
void st_close(SafeTensors *st);
size_t st_load(SafeTensors *st, const char *name,
               void *dst, size_t expected_nbytes);

#endif