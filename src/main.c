
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>

// #include "../include/utils.h"
//#include "../include/tokenizer.h"
// #include "../include/models/gpt.h"

//#include "../include/safetensors.h"
#include "../include/models/gpt2/gpt.h"
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

    // size_t vocab_size  = 50257;   // GPT-2 BPE vocab
    // size_t context_len = 7;    // Max sequence length (n_ctx)
    // size_t embed_dim   = 768;     // Embedding dimension (n_embd)
    // size_t n_heads     = 12;      // Attention heads (n_head)
    // size_t n_layers    = 12;      // Transformer blocks (n_layer)
    // float drop_rate    = 0.1;     // Dropout
    // bool qkv_bias      = true;    // GPT-2 uses bias in QKV
    // size_t batch_size  = 1;       // Batch Size 
    //char *params_filename = "/Users/uonliaquat/Downloads/gpt2.safetensors";

    size_t vocab_size  = 50257;   // GPT-2 BPE vocab
    size_t context_len = 7;    // Max sequence length (n_ctx)
    size_t embed_dim   = 1024;     // Embedding dimension (n_embd)
    size_t n_heads     = 16;      // Attention heads (n_head)
    size_t n_layers    = 24;      // Transformer blocks (n_layer)
    float drop_rate    = 0.1;     // Dropout
    bool qkv_bias      = true;    // GPT-2 uses bias in QKV
    size_t batch_size  = 1;       // Batch Size 

    char *params_filename = "/Users/uonliaquat/Downloads/gpt2-medium.safetensors";
    char *vocab_filename = "/Users/uonliaquat/workspace/zerograd/python/gpt2_vocab.bin";


    GPTModel model = model_gpt_init(
        params_filename, 
        vocab_filename, 
        vocab_size, 
        context_len, 
        embed_dim, 
        n_heads, 
        n_layers, 
        drop_rate, 
        qkv_bias, 
        batch_size, 
        DTYPE_FP32, 
        "transformer"
    );


    Tensor input_tokens = tensor_init(
        (int[]){
            8001, 9542, 4430,  287, 1160, 2075,  318
        }, 
        (uint32_t[]){1, 1, context_len}, 
        3, 
        DTYPE_INT32, 
        "input_tokens"
    );



    model_gpt_forward(&model, &input_tokens, "Artificial intelligence in 2026 is\0");
    // Tensor input_tokens = tensor_init(
    //     (int[]){
    //         818,    2274,   812,    11,     11666,  4430,   468,    1716,   281,    1593, 
    //         2891,   287,    867,    11798,  11,     5742,   4837,   16602,  1366,   11, 
    //         43511,  8861,   11,     290,    2987,    2551,  1642,   13,     10850,  4673, 
    //         4981,   389,    8776,   319,    1588,   40522,  290,    460,    7716,   2420,
    //         11,     7564,   7572,   11,     290,    4331,   10906,  1912,   319,    2180,
    //         6096,   13
    //     }, 
    //     (uint32_t[]){1, 1, 52}, 
    //     3, 
    //     DTYPE_INT32, 
    //     "input_tokens"
    // );

    //   Tensor input_tokens = tensor_init(
    //     (int[]){
    //         15496, 314, 1101, 257, 3303, 2746
    //     }, 
    //     (uint32_t[]){1, 1, 6}, 
    //     3, 
    //     DTYPE_INT32, 
    //     "input_tokens"
    // );

    // "Hello I'm a language model\0"


    // "In recent years, artificial intelligence has become an important tool "
    // "in many industries, helping researchers analyze data, automate tasks, "
    // "and improve decision making. Machine learning models are trained on "
    // "large datasets and can generate text, recognize patterns, and predict "
    // "outcomes based on previous examples.\0"
}

    //model_gpt_write(&model, "c_model.safetensors");

