
#include <string.h>

#include "../include/utils.h"
#include "../include/tokenizer.h"
#include "../include/dataset.h"
#include "../include/dataloader.h"
#include "../include/layers/embedding.h"



int main(){

    char *data = read_data_from_file("/Users/uonliaquat/workspace/zerograd/the-verdict.txt");
    size_t data_len = strlen(data);


    // printf("Data:\n%s\n\n", data);
    // printf("data_size: %zu\n", data_len);

    struct Vocab *vocab = tokenizer_init_vocab();
    struct Data *corpus = tokenizer_create_data(data);
    struct MergeRules *merge_rules = tokenizer_init_merge_rules();

    //tokenizer_train(corpus, vocab, merge_rules);

    //tokenizer_print_corpus(corpus);

    // tokenizer_write_vocab("./vocab.txt", vocab);
    // tokenizer_write_merge_rules("merge_rules.txt", merge_rules);



    tokenizer_read_vocab("/Users/uonliaquat/workspace/zerograd/vocab.txt", vocab);
    
    tokenizer_read_merge_rules("/Users/uonliaquat/workspace/zerograd/merge_rules.txt", merge_rules);
    
    // printf("Vocab Size: %zu\n", vocab->len);
    // for(size_t i = 0; i < vocab->len; i++){
    //     printf("%zu ",i);
    //     tokenizer_print_token(vocab->tokens[i]);
    //     printf("\n");
    // }

    // for(size_t i = 0; i < merge_rules_size; i++){
    //     print_byte_pair(merge_rules[i]);
    // }

    size_t ctx_win = 20;
    size_t stride = 15;


    Dataset dataset_gpt2 = dataset_build_gpt2(data, vocab, merge_rules , ctx_win, stride);
    dataset_write_gpt2(&dataset_gpt2, "./output/dataset_gpt.csv");

    EmbeddingLayer embeddings_table = embedding_layer_init(vocab->len, 32, false, DTYPE_DOUBLE);
    embedding_layer_write(&embeddings_table, "./output/embeddings_table.csv");
    

    DataLoader data_loader = dataloader_init(&dataset_gpt2, 1);
    DataSample data_sample = dataloader_get_next_batch(&data_loader);
    dataloader_print_sample(&data_sample);

    data_sample = dataloader_get_next_batch(&data_loader);
    dataloader_print_sample(&data_sample);

    data_sample = dataloader_get_next_batch(&data_loader);
    dataloader_print_sample(&data_sample);

    data_sample = dataloader_get_next_batch(&data_loader);
    dataloader_print_sample(&data_sample);
    

    tokenizer_free_vocab(vocab);
    tokenizer_free_merge_rules(merge_rules);

}