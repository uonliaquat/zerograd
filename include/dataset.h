#ifndef __DATASET_H__
#define __DATASET_H__

#include <stdio.h>
#include "../include/tensor.h"
#include "../include/tokenizer.h"

#define MAX_DATASET_SIZE 32000

typedef struct Dataset{
    Tensor x[MAX_DATASET_SIZE];
    Tensor y[MAX_DATASET_SIZE];
    size_t len;
} Dataset;

Dataset dataset_build_gpt2(char *data, Vocab *vocab, MergeRules *merge_rules, size_t ctx_win, size_t stride);
void dataset_print_gpt2(Dataset *dataset_gpt2);
void dataset_write_gpt2(Dataset *dataset_gpt, const char *filename);

#endif