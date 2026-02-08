
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

// #include "../include/utils.h"
//#include "../include/tokenizer.h"
// #include "../include/models/gpt.h"

#include "../include/safetensors.h"
#include "../include/models/gpt.h"
// #include "../include/dataset.h"
// #include "../include/dataloader.h"
// #include "../include/layers/embedding.h"
// #include "../include/layers/self_attention.h"

#include <assert.h>





int main(){


    // char *filename = "/Users/uonliaquat/Downloads/model.safetensors";
    //model_load_safetensors(filename);


    // char *data = read_data_from_file("/Users/uonliaquat/workspace/zerograd/dataset/train.txt");
    // size_t data_len = strlen(data);


    // //printf("Data:\n%s\n\n", data);
    // printf("data_size: %zu\n", data_len);

    // struct Vocab *vocab = tokenizer_init_vocab();
    // struct Data *corpus = tokenizer_create_data(data);
    // struct MergeRules *merge_rules = tokenizer_init_merge_rules();

    // tokenizer_train(corpus, vocab, merge_rules);

    // //tokenizer_print_corpus(corpus);

    // tokenizer_write_vocab("./vocab.txt", vocab);
    // tokenizer_write_merge_rules("merge_rules.txt", merge_rules);



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

    size_t vocab_size  = 50257;   // GPT-2 BPE vocab
    size_t context_len = 1024;    // Max sequence length (n_ctx)
    size_t embed_dim   = 768;     // Embedding dimension (n_embd)
    size_t n_heads     = 12;      // Attention heads (n_head)
    size_t n_layers    = 12;      // Transformer blocks (n_layer)
    float drop_rate    = 0.1;     // Dropout
    bool qkv_bias      = true;    // GPT-2 uses bias in QKV
    size_t batch_size  = 1;       // Batch Size 


    char *filename = "/Users/uonliaquat/Downloads/model.safetensors";
    GPTParams params;
    safetensors_load_model(filename, &params);
    char model_name[256] = "\0";
    strcpy(model_name, "gpt");
    GPTModel model = model_gpt_init(&params, vocab_size, context_len, embed_dim, n_heads, n_layers, drop_rate, qkv_bias, batch_size, DTYPE_FP32, model_name);
    
    Tensor input_tokens = tensor_init(
        (int[]){
            818,    2274,   812,    11,     11666,  4430,   468,    1716,   281,    1593, 
            2891,   287,    867,    11798,  11,     5742,   4837,   16602,  1366,   11, 
            43511,  8861,   11,     290,    2987,    2551,  1642,   13,     10850,  4673, 
            4981,   389,    8776,   319,    1588,   40522,  290,    460,    7716,   2420,
            11,     7564,   7572,   11,     290,    4331,   10906,  1912,   319,    2180,
            6096,   13
        }, 
        (uint32_t[]){1, 1, 52}, 
        3, 
        DTYPE_INT32, 
        "input_tokens"
    );
    //tensor_print(&input_tokens, "input_tokens");
    model_gpt_forward(&model, &input_tokens);
    model_gpt_write(&model, "c_model.safetensors");
    //model_gpt_safetensors_init(filename);

    // model_gpt_config_init(vocab_size, context_len, emb_dim, n_heads, n_layers, drop_rate, qkv_bias);

    // Tensor input_tokens = tensor_init((float[]){
    //     1, 2, 0, 2
    //     // 0, 0, 4, 3, 2, 1
    //     // 0.57, 0.85, 0.64, 0.64,
    //     // 0.22, 0.58, 0.33, 0.33,
    //     // 0.77, 0.25, 0.10, 0.10,
    //     // 0.05, 0.80, 0.55, 0.55,
    // }, (uint32_t[]){1, 1, 4}, 3, DTYPE_FP32, "input_tokens");
    //tensor_print(&input_tokens);
    //model_gpt_forward(&input_tokens);
    //model_gpt_free();


    // Dataset dataset_gpt2 = dataset_build_gpt2(data, vocab, merge_rules , seq_len, stride);
    // dataset_write_gpt2(&dataset_gpt2, "./output/dataset_gpt.csv");

    // DataLoader data_loader = dataloader_init(&dataset_gpt2, 1);
    //DataSample data_sample = dataloader_get_next_batch(&data_loader);
    // dataloader_print_sample(&data_sample);


    // Tensor token_embeddings = tensor_init((float[]){
    //     1, 2, 0, 2,
    //     0, 1, 2, 3,
    // }, (uint32_t[]){1, 2, 4}, 3, tensor_dtype_size(DTYPE_FP32), false, false);

    // tensor_print(&token_embeddings, );

    // EmbeddingLayer token_embed_layer = embedding_layer_init(vocab_size, emb_dim, DTYPE_FP32);
    // embedding_layer_forward(&token_embed_layer, &token_embeddings);
    // embedding_layer_print(&token_embed_layer);
    // embedding_layer_free(&token_embed_layer);
    // embedding_layer_write(&token_embed_layer, "./output/token_embedding_layer.csv");
    // tensor_print(&token_embeddings, );

    // EmbeddingLayer pos_embed_layer = embedding_layer_init(context_len, emb_dim, DTYPE_FP32);
    // embedding_layer_forward(&token_embed_layer, &token_embeddings);
    // embedding_layer_write(&token_embed_layer, "./output/token_embedding_layer.csv");
    // tensor_print(&token_embeddings, );
    
    // EmbeddingLayer pos_embedding_layer = embedding_layer_init(seq_len, embed_dim, seq_len, DTYPE_FP32);
    // Tensor embedding_layer_positional_output =embedding_layer_positional_forward(&pos_embedding_layer);
    // embedding_layer_write(&pos_embedding_layer, "./output/pos_embedding_layer.csv");
    // // // tensor_print(&embedding_layer_positional_output);

    // Tensor input_embeddings = tensor_add(&embedding_layer_token_output, &embedding_layer_positional_output);
    // tensor_print(&input_embeddings, );
    // // // tensor_write(&input_embeddings, "./output/input_embeddings.csv");

    // Tensor input_embeddings = tensor_init((float[]){
    //     0.43, 0.15, 0.89, 0.89,
    //     0.55, 0.87, 0.66, 0.66,
    //     0.57, 0.85, 0.64, 0.64,
    //     0.22, 0.58, 0.33, 0.33,
    //     0.77, 0.25, 0.10, 0.10,
    //     0.05, 0.80, 0.55, 0.55,
    // }, (uint32_t[]){seq_len, embed_dim}, 2, tensor_dtype_size(DTYPE_FP32), false, false);


    // // const size_t shape1[] = {seq_len, embed_dim};
    // // const size_t ndim1 = sizeof(shape1) / sizeof(size_t);
    // // Tensor tensor1 = tensor_init(NULL, shape1, ndim1, DTYPE_FP32, false, true);
    // // tensor_print(&tensor1, );
    // // tensor_write(&tensor1, "./output/tensor1.csv");

    // // const size_t shape_input_embeddings[] = {seq_len, embed_dim};
    // // const size_t ndim_input_embeddings = sizeof(shape_input_embeddings) / sizeof(size_t);
    // // Tensor input_embeddings = tensor_init(NULL, shape_input_embeddings, ndim_input_embeddings, DTYPE_FP32, false, true);
    // // tensor_print(&input_embeddings, );
    // // tensor_write(&input_embeddings, "./output/input_embeddings.csv");


    // SelfAttentionLayer self_attention_layer = self_attention_layer_init(seq_len, embed_dim, num_heads, false, false, DTYPE_FP32);
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