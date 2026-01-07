
#include <string.h>

#include "../include/utils.h"
#include "../include/tokenizer.h"
#include "../include/dataset.h"
#include "../include/layers/embedding.h"



int main(){

    char *data = read_data_from_file("/Users/uonliaquat/workspace/zerograd/the-verdict.txt");
    size_t data_len = strlen(data);


    // printf("Data:\n%s\n\n", data);
    // printf("data_size: %zu\n", data_len);

    // struct Vocab *vocab = tokenizer_init_vocab();
    // struct Data *corpus = tokenizer_create_data(data, vocab);



    // struct MergeRules *merge_rules = tokenizer_init_merge_rules();
    // tokenizer_train(corpus, vocab, merge_rules);

    // //tokenizer_print_corpus(corpus);

    // tokenizer_write_vocab("./vocab.txt", vocab);
    // tokenizer_write_merge_rules("merge_rules.txt", merge_rules);




    Vocab *vocab = tokenizer_init_vocab();
    tokenizer_read_vocab("/Users/uonliaquat/workspace/zerograd/vocab.txt", vocab);
    // printf("Vocab Size: %zu\n", vocab_size);
    // // for(size_t i = 0; i < vocab_size; i++){
    // //     print_token(vocab[i]);
    // //     printf("\n");
    // // }

    MergeRules *merge_rules = tokenizer_init_merge_rules();
    tokenizer_read_merge_rules("/Users/uonliaquat/workspace/zerograd/merge_rules.txt", merge_rules);
    // // for(size_t i = 0; i < merge_rules_size; i++){
    //     print_byte_pair(merge_rules[i]);
    // }


    DatasetGPT2 dataset_gpt2 = dataset_build_gpt2(data, vocab, merge_rules, 20, 15);
    dataset_write_gpt2(&dataset_gpt2, "dataset_gpt.csv");
    //dataset_print_dataset_gpt2(&dataset_gpt2);

    // EmbeddingLayer embeddings_table = embedding_layer_init(100, 32, false, DTYPE_DOUBLE);
    // embedding_layer_write(&embeddings_table, "embedding_table.csv");
    

    tokenizer_free_vocab(vocab);
    tokenizer_free_merge_rules(merge_rules);

}