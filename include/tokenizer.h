#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include <stdio.h>
#include <stdint.h>

#define MAX_TOKEN_LEN 64
#define MAX_VOCAB_SIZE 60000

#define MAX_WORD_LEN 50
#define MAX_UNIQUE_WRODS 200000
#define MAX_BYTE_PAIRS 500000

#define STX 2
#define ETX 3


typedef struct Token{
    char token[MAX_TOKEN_LEN];
    //int16_t token_id;
} Token;

typedef struct Word{
    int16_t token_ids[MAX_WORD_LEN];
    int32_t freq;
} Word;

typedef struct Corpus{
    Word words[MAX_UNIQUE_WRODS];
    int32_t len;
} Corpus;

typedef struct BytePair{
    int16_t token1_id;
    int16_t token2_id;
    int32_t freq;
} BytePair;

typedef struct BytePairs{
    BytePair pairs[MAX_BYTE_PAIRS];
    int32_t len;
}BytePairs;

typedef struct Vocab{
    Token tokens[MAX_VOCAB_SIZE];
    int32_t len;
} Vocab;

void tokenizer_read_vocab(const char *filename, Vocab *vocab);


#endif