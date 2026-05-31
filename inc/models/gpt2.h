#ifndef __MODEL_GPT2_H__
#define __MODEL_GPT2_H__

#include <string.h>
#include <stdbool.h>

typedef struct GPT2Config{
    size_t ctx_win;
    size_t vocab_size;
    size_t ndim;
    size_t nheads;
    size_t nlayers;
    bool qkv_bias;
} GPT2Config;

typedef struct GPT2Offsets{
    size_t token_ids;
    size_t wpe, wte;
    size_t token_embed, pos_emebd;
    size_t input_embed;
    size_t ln1;
} GPT2Offsets;


void build_gpt2(GPT2Config *config);
#endif
