#include "../include/dataloader.h"
// #include "../include/tokenizer.h"


DataLoader dataloader_init(Dataset *dataset, size_t batch_size){
    DataLoader data_loader;
    data_loader.dataset = dataset;
    data_loader.batch_size = batch_size;
    data_loader.curr_sample = 0;
    return data_loader;
}

DataSample dataloader_get_next_batch(DataLoader *data_loader){
    DataSample data_sample;
    data_sample.x = &data_loader->dataset->x[data_loader->curr_sample];
    data_sample.y = &data_loader->dataset->y[data_loader->curr_sample];
    data_sample.idx = data_loader->curr_sample;
    data_loader->curr_sample++;
    return data_sample;
}

void dataloader_print_sample(DataSample *data_sample){
    printf("\n====================SAMPLE %zu ==================\n", data_sample->idx);
    tensor_print(data_sample->x, "X");
    tensor_print(data_sample->y, "Y");
    printf("========================================= ========\n");
}