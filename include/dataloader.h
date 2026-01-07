#ifndef __DATALOADER_H__
#define __DATALOADER_H_


#include "./dataset.h"
typedef struct DataLoader{
    void *dataset;
    size_t batch_size;
    size_t curr_sample;

} DataLoader;

DataLoader dataloader_init(void *dataset, size_t batch_size);
void* dataloader_get_next_batch();


// #endif