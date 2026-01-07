#include "../include/dataloader.h"
// #include "../include/tokenizer.h"


DataLoader dataloader_init(void *dataset, size_t batch_size){
    DataLoader data_loader;
    data_loader.dataset = dataset;
    data_loader.batch_size = batch_size;
}

void *dataloader_get_next_batch(DataLoader *data_loader){
    ((DatasetGPT2*)data_loader)->

    //data_loader->curr_sample++;
}