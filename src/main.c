
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

    size_t ctx_win = 10;
    size_t stride = 15;
    size_t embed_len = 10;
    size_t vocab_size = vocab->len;


    Dataset dataset_gpt2 = dataset_build_gpt2(data, vocab, merge_rules , ctx_win, stride);
    dataset_write_gpt2(&dataset_gpt2, "./output/dataset_gpt.csv");

    DataLoader data_loader = dataloader_init(&dataset_gpt2, 1);
    DataSample data_sample = dataloader_get_next_batch(&data_loader);
    //dataloader_print_sample(&data_sample);


    EmbeddingLayer token_embedding_layer = embedding_layer_init(vocab_size, embed_len, ctx_win, DTYPE_DOUBLE);
    Tensor embedding_layer_token_output = embedding_layer_token_forward(&token_embedding_layer, data_sample.x);
    //embedding_layer_write(&token_embedding_layer, "./output/token_embedding_layer.csv");
    tensor_print(&embedding_layer_token_output);
    
    EmbeddingLayer pos_embedding_layer = embedding_layer_init(ctx_win, embed_len, ctx_win, DTYPE_DOUBLE);
    Tensor embedding_layer_positional_output =embedding_layer_positional_forward(&pos_embedding_layer);
    //embedding_layer_write(&pos_embedding_layer, "./output/pos_embedding_layer.csv");
    tensor_print(&embedding_layer_positional_output);

    // Tensor input_embeddings = tensor_add(&embedding_layer_token_output, &embedding_layer_positional_output);
    // tensor_print(&input_embeddings);
    // tensor_write(&input_embeddings, "./output/input_embeddings.csv");

    Tensor input_embeddings = tensor_init((double[]){
        0.43, 0.15, 0.89,
        0.55, 0.87, 0.66,
        0.57, 0.85, 0.64,
        0.22, 0.58, 0.33,
        0.77, 0.25, 0.10,
        0.05, 0.80, 0.55    }, (size_t[]){6,3}, 2, sizeof(double), false, false);   
    tensor_print(&input_embeddings);
    tensor_write(&input_embeddings, "./output/input_embeddings.csv");

    Tensor input_embeddings_transposed = tensor_transpose(&input_embeddings);
    tensor_print(&input_embeddings_transposed);
    tensor_write(&input_embeddings_transposed, "./output/input_embeddings_transposed.csv");

    Tensor attention_scores = tensor_dot_product_matrix(&input_embeddings, &input_embeddings_transposed);
    tensor_print(&attention_scores);

    Tensor attention_weights = tensor_softmax(&attention_scores, 1);
    tensor_print(&attention_weights);

    Tensor context_vectors = tensor_dot_product_matrix(&attention_weights, &input_embeddings);
    tensor_print(&context_vectors);

    


    tokenizer_free_vocab(vocab);
    tokenizer_free_merge_rules(merge_rules);

}