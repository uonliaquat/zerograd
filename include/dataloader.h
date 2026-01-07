#ifndef __DATALOADER_H__
#define __DATALOADER_H__


#include "./dataset.h"
typedef struct DataLoader{
    Dataset *dataset;
    size_t batch_size;
    size_t curr_sample;

} DataLoader;

typedef struct DataSample{
    Tensor *x;
    Tensor *y;
    size_t idx;
} DataSample;

DataLoader dataloader_init(Dataset *dataset, size_t batch_size);
DataSample dataloader_get_next_batch(DataLoader *data_loader);
void dataloader_print_sample(DataSample *data_sample);


#endif