
#include <string.h>

#include "../include/tensor.h"
#include "../include/utils.h"
#include "../include/tokenizer.h"
#include "../include/dataset.h"
#include "../include/dataloader.h"
#include "../include/layers/embedding.h"
#include "../include/layers/self_attention.h"
#include "../include/layers/self_attention.h"



int main(){

    char *data = read_data_from_file("/Users/uonliaquat/workspace/zerograd/dataset/train.txt");
    size_t data_len = strlen(data);


    printf("Data:\n%s\n\n", data);
    printf("data_size: %zu\n", data_len);

    struct Vocab *vocab = tokenizer_init_vocab();
    struct Data *corpus = tokenizer_create_data(data);
    struct MergeRules *merge_rules = tokenizer_init_merge_rules();

    tokenizer_train(corpus, vocab, merge_rules);

    //tokenizer_print_corpus(corpus);

    tokenizer_write_vocab("./vocab.txt", vocab);
    tokenizer_write_merge_rules("merge_rules.txt", merge_rules);



    // tokenizer_read_vocab("/Users/uonliaquat/workspace/zerograd/vocab.txt", vocab);
    
    // tokenizer_read_merge_rules("/Users/uonliaquat/workspace/zerograd/merge_rules.txt", merge_rules);
    
    // printf("Vocab Size: %zu\n", vocab->len);
    // for(size_t i = 0; i < vocab->len; i++){
    //     printf("%zu ",i);
    //     tokenizer_print_token(vocab->tokens[i]);
    //     printf("\n");
    // }

    // for(size_t i = 0; i < merge_rules_size; i++){
    //     print_byte_pair(merge_rules[i]);
    // }

    // size_t batch_size = 4;
    // size_t seq_len = 6;
    // size_t stride = 15;
    // size_t embed_dim = 4;
    // size_t num_heads = 2;
    // size_t vocab_size = vocab->len;

    // Dataset dataset_gpt2 = dataset_build_gpt2(data, vocab, merge_rules , seq_len, stride);
    // dataset_write_gpt2(&dataset_gpt2, "./output/dataset_gpt.csv");

    // DataLoader data_loader = dataloader_init(&dataset_gpt2, 1);
    // DataSample data_sample = dataloader_get_next_batch(&data_loader);
    // dataloader_print_sample(&data_sample);


    // EmbeddingLayer token_embedding_layer = embedding_layer_init(vocab_size, embed_dim, seq_len, DTYPE_DOUBLE);
    // Tensor embedding_layer_token_output = embedding_layer_token_forward(&token_embedding_layer, data_sample.x);
    // embedding_layer_write(&token_embedding_layer, "./output/token_embedding_layer.csv");
    // // //tensor_print(&embedding_layer_token_output);
    
    // EmbeddingLayer pos_embedding_layer = embedding_layer_init(seq_len, embed_dim, seq_len, DTYPE_DOUBLE);
    // Tensor embedding_layer_positional_output =embedding_layer_positional_forward(&pos_embedding_layer);
    // embedding_layer_write(&pos_embedding_layer, "./output/pos_embedding_layer.csv");
    // // // tensor_print(&embedding_layer_positional_output);

    // Tensor input_embeddings = tensor_add(&embedding_layer_token_output, &embedding_layer_positional_output);
    // tensor_print(&input_embeddings, "Input Emeddings");
    // // // tensor_write(&input_embeddings, "./output/input_embeddings.csv");

    // // Tensor input_embeddings = tensor_init((double[]){
    // //     0.43, 0.15, 0.89, 0.89,
    // //     0.55, 0.87, 0.66, 0.66,
    // //     0.57, 0.85, 0.64, 0.64,
    // //     0.22, 0.58, 0.33, 0.33,
    // //     0.77, 0.25, 0.10, 0.10,
    // //     0.05, 0.80, 0.55, 0.55,
    // // }, (size_t[]){seq_len, embed_dim}, 2, tensor_dtype_size(DTYPE_DOUBLE), false, false);


    // // const size_t shape1[] = {seq_len, embed_dim};
    // // const size_t ndim1 = sizeof(shape1) / sizeof(size_t);
    // // Tensor tensor1 = tensor_init(NULL, shape1, ndim1, DTYPE_DOUBLE, false, true);
    // // tensor_print(&tensor1, "Tensor 1");
    // // tensor_write(&tensor1, "./output/tensor1.csv");

    // // const size_t shape_input_embeddings[] = {seq_len, embed_dim};
    // // const size_t ndim_input_embeddings = sizeof(shape_input_embeddings) / sizeof(size_t);
    // // Tensor input_embeddings = tensor_init(NULL, shape_input_embeddings, ndim_input_embeddings, DTYPE_DOUBLE, false, true);
    // // tensor_print(&input_embeddings, "Input Embeddings");
    // // tensor_write(&input_embeddings, "./output/input_embeddings.csv");


    // SelfAttentionLayer self_attention_layer = self_attention_layer_init(seq_len, embed_dim, num_heads, false, false, DTYPE_DOUBLE);
    // self_attention_layer_print(&self_attention_layer, "Multi Head Self Attention");
    // self_attention_layer_write(&self_attention_layer, "./output/self_attention_layer.csv");
    // //Tensor context_vecs = self_attention_layer_forward(&self_attention_layer, &input_embeddings);
    // // // tensor_print(&context_vecs, "context vecs");

    // Tensor context_vecs = self_attention_layer_mult_head_forward(&self_attention_layer, &input_embeddings);
    // // tensor_print(&context_vecs, "context vecs");

    // Tensor context_vecs = self_attention_layer_forward(&self_attention_layer, &input_embeddings);
    // //tensor_print(&context_vecs, "Context Vectors");

    // tokenizer_free_vocab(vocab);
    // tokenizer_free_merge_rules(merge_rules);

}