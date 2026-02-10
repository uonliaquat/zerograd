#ifndef __GPT_H__
#define __GPT_H__

#include<stdbool.h>

#include "./tensor.h"
#include "./layers/embedding.h"
#include "./layers/transformer.h"
#include "../include/tokenizer.h"
// #include "../layers/dropout.h"
// #include "../layers/linear.h"
// #include "../layers/layer_norm.h"
// #include "../layers/transformer.h"




typedef struct GPTConfig {
    size_t vocab_size;      // Vocab Size
    size_t context_len;     // Context Length
    size_t embed_dim;       // Embedding Dimensions
    size_t n_heads;         // No of Attention heads
    size_t n_layers;        // No of layers
    float drop_rate;       // Dropout rate
    bool qkv_bias;          // Query-Key-Value bias
    size_t batch_size;
    DataType dtype;
    char name[256];
} GPTConfig;





typedef struct GPTParams{
    EmbeddingLayerParams wpe;
    EmbeddingLayerParams wte;
    TransformerLayerParams *h;
    LayerNormParams ln_f;
    LinearLayerParams head;
} GPTParams;


// typedef struct GPTParams{
//     struct {
//         Tensor weight;
//     } wpe;
//     struct{
//         Tensor weight;
//     } wte;
//     struct {
//         struct {
//             Tensor bias;
//             struct {
//                 Tensor bias;
//                 Tensor weight;
//             } c_attn;
//             struct {
//                 Tensor bias;
//                 Tensor weight;
//             } c_proj;
//         } attn;
//         struct {
//             struct {
//                 Tensor bias;
//                 Tensor weight;
//             } c_fc;
//             struct {
//                 Tensor bias;
//                 Tensor weight;
//             } c_proj;
//         } mlp;
//         struct {
//             Tensor bias;
//             Tensor weight;
//         } ln[2];
//     } h[12];
//     struct {
//         Tensor bias;
//         Tensor weight;
//     } ln_f;
//     struct {
//         Tensor bias;
//         Tensor weight;
//     } head;

// } GPTParams;

typedef struct GPTWrokspace{
    Tensor indices;
    Tensor pos_indices;
    Tensor *embeddings;
    Tensor next_token_prob_dist;
    Tensor output;
} GPTWrokspace;


typedef struct GPTModel{
    char name[128];
    GPTConfig config;
    GPTParams params;
    Vocab vocab;
    EmbeddingLayer wte;
    EmbeddingLayer wpe;
    TransformerLayer *h;
    LayerNorm ln_f;
    LinearLayer head;
    GPTWrokspace workspace;
} GPTModel;


GPTModel model_gpt_init(
    const char *params_filename,
    const char *vocab_filename,
    const size_t vocab_size,
    const size_t context_len, 
    const size_t embed_dim, 
    const size_t n_heads, 
    const size_t n_layers, 
    const float drop_rate, 
    const bool qkv_bias, 
    const size_t batch_size,
    const DataType dtype,
    char *name
);

void model_gpt_free(GPTModel *model);
void model_gpt_forward(GPTModel *model, Tensor *x, const char *prompt);
void model_gpt_write(GPTModel *model, const char *filename);
void model_gpt_init_params(const char *filename, GPTModel *model);
// void model_gpt_safetensors_write(const char *filename, GPTParameters *params);



//void model_gpt_write(const char *path);


#endif