#include "../include/dataset.h"


#include <stdlib.h>
#include <string.h>


DatasetGPT2 dataset_build_gpt2(char *data, Vocab *vocab, MergeRules *merge_rules, size_t ctx_win, size_t stride){
    size_t token_ids_len = 0;
    int token_ids[MAX_WORDS];

    tokenizer_encode(data, token_ids, &token_ids_len, vocab, merge_rules);

    // for(size_t i = 0; i < token_ids_len; i++){
    //     printf("%d ", token_ids[i]);
    // }
    DatasetGPT2 dataset_gpt2;
    dataset_gpt2.len = 0;
    for(size_t i = 0; i < token_ids_len - 1; i = i + stride){
        int *x = calloc(ctx_win, sizeof(int));
        int *y = calloc(ctx_win, sizeof(int));
        memcpy(x, token_ids + i, sizeof(int)*ctx_win);
        memcpy(y, token_ids + i + 1, sizeof(int)*ctx_win);
        Tensor tensor_x = tensor_init(x, (size_t[]){ctx_win}, 1, DTYPE_INT, false, false);
        Tensor tensor_y = tensor_init(y, (size_t[]){ctx_win}, 1, DTYPE_INT, false, false);
        dataset_gpt2.x[dataset_gpt2.len] = tensor_x;
        dataset_gpt2.y[dataset_gpt2.len] = tensor_y;
        dataset_gpt2.len++;
        free(x);
        free(y);
    }

    return dataset_gpt2;
}

void dataset_print_gpt(DatasetGPT2 *dataset_gpt2){
    for(size_t i = 0; i < dataset_gpt2->len; i++){
        printf("\n========================================DATASET==============================================\n");
        tensor_print(&dataset_gpt2->x[i]);
        tensor_print(&dataset_gpt2->y[i]);
        printf("\n===============================================================================================\n\n");
    }
}

void dataset_write_gpt2(DatasetGPT2 *dataset_gpt2, const char *filename){
    FILE *fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    fprintf(fptr, "Inputs:\n");
    for(size_t i = 0; i < dataset_gpt2->len; i++){
        tensor_write(&dataset_gpt2->x[i], fptr);
    }
    fprintf(fptr, "\n\nTargets:\n");
    for(size_t i = 0; i < dataset_gpt2->len; i++){
        tensor_write(&dataset_gpt2->y[i], fptr);
    }
    fclose(fptr);
}