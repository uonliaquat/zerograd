#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include <stdio.h>


#define MAX_WORDS 256000
#define MAX_BYTE_PAIRS 256000
#define MAX_MERGE_RULES 32000
#define MAX_VOCAB_SIZE 32000
#define STX 2
#define ETX 3


typedef struct Token{
    char *val;
    size_t len;
} Token;

typedef struct Word{
    struct Token **tokens;
    size_t len;
    size_t freq;
} Word;

typedef struct Data{
    struct Word **words;
    size_t len;
} Data;

typedef struct BytePair{
    struct Token *token1;
    struct Token *token2;
    size_t freq;
} BytePair;

typedef struct Vocab{
    struct Token **tokens;
    size_t len;
} Vocab;

typedef struct MergeRules{
    struct BytePair **byte_pairs;
    size_t len;
} MergeRules;



Token *tokenizer_create_token(char *data, size_t data_size);
size_t tokenizer_match_tokens(Token *token1, Token *token2);
int tokenizer_get_token_index(Vocab *vocab, Token *token);
Token *tokenizer_concat_tokens(Token *token1, Token *token2);
void tokenizer_free_token(Token *token);
void tokenizer_print_token(Token *token);


Word *tokenizer_create_word(Token **tokens, size_t len);
size_t tokenizer_match_words(Word *word1, Word *word2);
int tokenizer_get_word_index(Word **words, size_t no_of_words, Word *word);
void tokenizer_free_word(Word *word);
void tokenizer_print_word(Word *word);

BytePair *tokenizer_create_byte_pair(Token *token1, Token *token2, size_t freq);
int tokenizer_get_byte_pair_index(BytePair **pairs, size_t no_of_byte_pairs, BytePair *byte_pair);
BytePair *get_most_frequent_byte_pair(BytePair **byte_pairs, size_t no_of_byte_pairs);
size_t tokenizer_get_byte_pairs(Data *data, BytePair **byte_pairs);
void tokenizer_free_byte_pairs(BytePair **byte_pairs, size_t no_of_byte_pairs);
void tokenizer_print_byte_pairs(BytePair **byte_pairs, size_t no_of_byte_pairs);
void tokenizer_free_byte_pair(BytePair *byte_pair);
void tokenizer_print_byte_pair(BytePair *byte_pair);


Data *tokenizer_create_data(char *data);
void tokenizer_free_data(Data *data);
void tokenizer_print_data(Data *data);



void tokenizer_merge(Data *data, BytePair *most_frequent_byte_pair);
int tokenizer_get_merge_rule_index(MergeRules *merge_rules, Token *token1, Token *token2);

Vocab *tokenizer_init_vocab();
void tokenizer_free_vocab(Vocab *vocab);
void tokenizer_write_vocab(char *filename, Vocab *vocab);
void tokenizer_read_vocab(char *filename, Vocab *vocab);

MergeRules *tokenizer_init_merge_rules();
void tokenizer_free_merge_rules(MergeRules *merge_rules);
void tokenizer_write_merge_rules(char *file_name, MergeRules *merge_rules);
void tokenizer_read_merge_rules(char *filename, MergeRules *merge_rules);

void tokenizer_train(Data *data, Vocab *vocab, MergeRules *merge_rules);
void tokenizer_encode(char *text, int *token_ids, size_t *token_ids_len, Vocab *vocab, MergeRules *merge_rules);
void tokenizer_decode(int *token_ids, Vocab *vocab);

#endif










