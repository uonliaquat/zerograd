#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

#include "../include/tokenizer.h"
#include "../include/utils.h"



Token *tokenizer_create_token(char *data, size_t data_size){
    Token *new_token = calloc(1, sizeof(Token));
    new_token->len = data_size;
    new_token->val = calloc(data_size, sizeof(char));
    memcpy(new_token->val, data, data_size);
    return new_token;
}

size_t tokenizer_match_tokens(Token *token1, Token *token2){
    if(token1->len != token2->len) return 0;
    int res = strcmp(token1->val, token2->val);
    if (res == 0) return 1;
    return 0;
}

int tokenizer_get_token_index(Vocab *vocab, Token *token){
    for(size_t i = 0; i < vocab->len; i++){
        // tokenizer_print_token(vocab->tokens[i]);
        // printf("   |  ");
        // tokenizer_print_token(token);
        // printf(" | %zu,   %zu  \n", vocab->tokens[i]->len, token->len);
        int res = tokenizer_match_tokens(vocab->tokens[i], token);
        if(res == 1) return i;
    }
    printf("\nToken not found in vocab\n");
    tokenizer_print_token(token);
    exit(1);
    return -1;
}

Token *tokenizer_concat_tokens(Token *token1, Token *token2){
    // size_t new_token_len = token1->len + token2->len + 1;
    // Token *new_token = tokenizer_create_token(token1->val, new_token_len);
    // memcpy(new_token->val + token1->len, token2->val, token2->len);
    // return new_token;

    Token *new_token = calloc(1, sizeof(Token));
    new_token->len = token1->len + token2->len;
    new_token->val = calloc(new_token->len+1, sizeof(char));
    memcpy(new_token->val, token1->val, token1->len);
    memcpy(new_token->val + token1->len, token2->val, token2->len);
    return new_token;
}

void tokenizer_free_token(Token *token){
    if(token->len > 0){
        free(token->val);
        token->len = 0;
    }
}

void tokenizer_print_token(Token *token){
    printf("[%s] | len: %u", token->val, token->len);
}

Word *tokenizer_create_word(Token **tokens, size_t len){
    Word *new_word = calloc(1, sizeof(Word));
    new_word->len = len;
    new_word->tokens = tokens;
    new_word->freq = 1;
    return new_word;
}

size_t tokenizer_match_words(Word *word1, Word *word2){
    if(word1->len != word2->len) return 0;
    for(size_t i = 0; i < word1->len; i++){
        size_t res = tokenizer_match_tokens(word1->tokens[i], word2->tokens[i]);
        if(res == 0) return res;
    }
    return 1;
}

int tokenizer_get_word_index(Word **words, size_t no_of_words, Word *word){
    for(size_t i = 0; i < no_of_words; i++){
        int res = tokenizer_match_words(words[i], word);
        if(res == 1) return i;
    }
    return -1;
}

void tokenizer_free_word(Word *word){
    for(size_t i = 0; i < word->len; i++){
        tokenizer_free_token(word->tokens[i]);
    }
    //free(word);
    word->len = 0;
    word->freq = 0;
}

void tokenizer_print_word(Word *word){
    for(size_t i = 0; i < word->len; i++){
        tokenizer_print_token(word->tokens[i]);
    }
    printf(" | len: %u | freq: %zu\n", word->len, word->freq);
}


BytePair *tokenizer_create_byte_pair(Token *token1, Token *token2, size_t freq){
    BytePair *new_byte_pair = calloc(1, sizeof(BytePair));
    new_byte_pair->token1 = tokenizer_create_token(token1->val, token1->len);
    new_byte_pair->token2 = tokenizer_create_token(token2->val, token2->len);
    new_byte_pair->freq = freq;
    return new_byte_pair;
}


int tokenizer_get_byte_pair_index(BytePair **pairs, size_t no_of_byte_pairs, BytePair *byte_pair){
    for(size_t i = 0; i < no_of_byte_pairs; i++){
       int res1 = tokenizer_match_tokens(pairs[i]->token1, byte_pair->token1);
       int res2 = tokenizer_match_tokens(pairs[i]->token2, byte_pair->token2);
       if(res1 == 1 && res2 == 1) return i;
    }
    return -1;
}

BytePair *tokenizer_get_most_frequent_byte_pair(BytePair **byte_pairs, size_t no_of_byte_pairs){
    BytePair *most_frequent_byte_pair = tokenizer_create_byte_pair(byte_pairs[0]->token1, byte_pairs[0]->token2, byte_pairs[0]->freq);
    for(size_t i = 1; i <no_of_byte_pairs; i++){
        if(byte_pairs[i]->freq > most_frequent_byte_pair->freq){
            tokenizer_free_byte_pair(most_frequent_byte_pair);
            most_frequent_byte_pair = tokenizer_create_byte_pair(byte_pairs[i]->token1, byte_pairs[i]->token2, byte_pairs[i]->freq);
        }
    }
    return most_frequent_byte_pair;
}

size_t tokenizer_get_byte_pairs(Data *data, BytePair **byte_pairs){
    size_t no_of_byte_pairs = 0;
    for(size_t i = 0; i < data->len; i++){
        Word *word = data->words[i];
        for(size_t j = 1; j < word->len; j++){
            Token *token_prev = word->tokens[j-1];
            Token *token_curr = word->tokens[j];
            BytePair *byte_pair = tokenizer_create_byte_pair(token_prev, token_curr, 1);
            size_t byte_pair_index = tokenizer_get_byte_pair_index(byte_pairs, no_of_byte_pairs, byte_pair);
            if(byte_pair_index == -1){
                byte_pairs[no_of_byte_pairs++] = byte_pair;
            }
            else{
                byte_pairs[byte_pair_index]->freq += word->freq;
            }
        }
    }
    return no_of_byte_pairs;
}

void tokenizer_free_byte_pairs(BytePair **byte_pairs, size_t no_of_byte_pairs){
    for(size_t i = 0; i < no_of_byte_pairs; i++){
        tokenizer_free_byte_pair(byte_pairs[i]);
    }
    no_of_byte_pairs = 0;
}

void tokenizer_print_byte_pairs(BytePair **byte_pairs, size_t no_of_byte_pairs){
    printf("Byte Pairs:\n");
    for(size_t i = 0; i < no_of_byte_pairs; i++){
        tokenizer_print_byte_pair(byte_pairs[i]);
    }
}

void tokenizer_free_byte_pair(BytePair *byte_pair){
    tokenizer_free_token(byte_pair->token1);
    tokenizer_free_token(byte_pair->token2);
    byte_pair->freq = 0;
}

void tokenizer_print_byte_pair(BytePair *byte_pair){
    tokenizer_print_token(byte_pair->token1);
    tokenizer_print_token(byte_pair->token2);
    printf(" | freq: %zu\n", byte_pair->freq);
}

int tokenizer_is_ascii_string(const char *s) {
    for (size_t i = 0; i < strlen(s); i++) {
        if ((unsigned char)s[i] > 127) {
            return 0; // non-ASCII, skip
        }
    }
    return 1; // all ASCII
}

Data *tokenizer_create_data(char *data){
    printf("\n ******************* Creating Data ******************* \n");
    size_t data_size = strlen(data);

    Word **words = calloc(MAX_WORDS, sizeof(Word*));
    size_t total_unique_words = 0;
    char curr_word[MAX_WORD_LEN] = {0};
    size_t curr_word_index = 0;
    for(size_t i = 0; i < data_size; i++){
        printf("%zu | %zu | %s \n", i, curr_word_index, curr_word);
        char c =  tolower((unsigned char)data[i]);
        curr_word[curr_word_index++] = c;
        if (c == ' '    || c == '\n'    || c == '\t'    || c == '!' ||
            c == ';'    || c == ','     || c == ':'     || c == '"' ||
            c == '\''   || c == '.'     || c == '-'     || c == '?' ){

                if(curr_word_index > MAX_WORD_LEN || tokenizer_is_ascii_string(curr_word) == 0){
                    curr_word_index = 0;
                    memset(curr_word, 0, MAX_WORD_LEN);
                };

                if(i%1000000 == 0) printf("total_unique_words %zu\n", total_unique_words);
            
                Token **tokens = calloc(curr_word_index, sizeof(Token*));
                for(size_t j = 0; j < curr_word_index; j++){
                    Token *new_token = tokenizer_create_token(&curr_word[j], 1);
                    tokens[j] = new_token;
                }
                Word *new_word = tokenizer_create_word(tokens, curr_word_index);
                int word_index = tokenizer_get_word_index(words, total_unique_words, new_word);
                if(word_index == -1){
                    words[total_unique_words++] = new_word;
                }
                else{
                    words[word_index]->freq++;
                    tokenizer_free_word(new_word);
                }
                curr_word_index = 0;
                memset(curr_word, 0, MAX_WORD_LEN);
        }
        if(curr_word_index >= 30) printf("\n%s\n", curr_word);
    }
    Data *new_data = calloc(1, sizeof(Data));
    new_data->words = words;
    new_data->len = total_unique_words;
    printf("******************* Data Created ****************** \n");
    return new_data;
}


void tokenizer_free_data(Data *data){
    for(size_t i = 0; i < data->len; i++){
        tokenizer_free_word(data->words[i]);
    }
    data->len = 0;
}

void tokenizer_print_data(Data *data){
    printf("No of Unique Words in Data: %zu\n", data->len);
    for(size_t i = 0; i < data->len; i++){
        tokenizer_print_word(data->words[i]);
    }
}


void tokenizer_merge(Data *data, BytePair *most_frequent_byte_pair){
    for(size_t i = 0; i < data->len; i++){
        Word *word = data->words[i];
        for(size_t j = 1; j < word->len; j++){
            Token *token1 = word->tokens[j-1];
            Token *token2 = word->tokens[j];
            int res1 = tokenizer_match_tokens(token1, most_frequent_byte_pair->token1);
            int res2 = tokenizer_match_tokens(token2, most_frequent_byte_pair->token2);
            if(res1 == 1 && res2 == 1){
                Token *new_token = tokenizer_concat_tokens(token1, token2);
                tokenizer_free_token(token1);
                tokenizer_free_token(token2);
                word->tokens[j-1] = new_token;
                for(size_t k = j+1; k < word->len; k++){
                    word->tokens[k-1] =  word->tokens[k];
                }
                word->len--;
            }
        }
    }
}

int tokenizer_get_merge_rule_index(MergeRules *merge_rules, Token *token1, Token *token2){
    for(size_t i = 0; i < merge_rules->len; i++){
        int res1 = tokenizer_match_tokens(merge_rules->byte_pairs[i]->token1, token1);
        int res2 = tokenizer_match_tokens(merge_rules->byte_pairs[i]->token2, token2);
        if(res1 == 1 && res2 == 1) return i;
    }
    return -1;
}

Vocab *tokenizer_init_vocab(){
    printf("\n***************** Initializing Vocab ***************** \n");
    Vocab *vocab = calloc(1, sizeof(Vocab));
    vocab->tokens = calloc(MAX_VOCAB_SIZE, sizeof(Token*));
    vocab->len = 0;
    printf("***************** Vocab Initialized ****************** \n");
    return vocab;
}

void tokenizer_free_vocab(Vocab *vocab){
    for(size_t i = 0; i < vocab->len; i++){
        tokenizer_free_token(vocab->tokens[i]);
    }
    vocab->len = 0;
}

void tokenizer_write_vocab(char *filename, Vocab *vocab){
    printf("\n\n---------------Writing Vocab------------------\n");
    FILE *fptr = fopen(filename, "w");  // fresh file
    fclose(fptr);
    fptr = fopen(filename, "a");
    for(size_t i = 0; i < vocab->len; i++){
        fprintf(fptr, "%c%s%c\n", 
            STX, 
            vocab->tokens[i]->val,
            ETX
        );
    }
    fclose(fptr);
    printf("-------------Done Writing Vocab----------------\n");
}

void tokenizer_read_vocab(char *filename, Vocab *vocab){
    printf("\n\n---------------Reading Vocab------------------\n");

    char *data = read_data_from_file(filename);
    size_t data_size = strlen(data);

    char token_text[48] = {0};
    size_t token_text_index = 0;
    Token *token = NULL;
    bool read = false;
    for(size_t i = 0; i < data_size; i++){
        if(data[i] == STX){
            read = true;
            memset(token_text, 0, strlen(token_text));
            token_text_index = 0;
        }
        else if(data[i] == ETX){
            read = false;
            token = tokenizer_create_token(token_text, token_text_index);
            vocab->tokens[vocab->len++] = token;
            
        }
        else if(read){
            token_text[token_text_index++] = data[i];
        }
    }
    printf("-------------Done Reading Vocab----------------\n");
}

MergeRules *tokenizer_init_merge_rules(){
    MergeRules *merge_rules = calloc(1, sizeof(MergeRules));
    merge_rules->byte_pairs = calloc(MAX_MERGE_RULES, sizeof(BytePair*));
    merge_rules->len = 0;
    return merge_rules;
}

void tokenizer_free_merge_rules(MergeRules *merge_rules){
    for(size_t i = 0; i < merge_rules->len; i++){
        tokenizer_free_byte_pair(merge_rules->byte_pairs[i]);
    }
    merge_rules->len = 0;
}

void tokenizer_write_merge_rules(char *filename, MergeRules *merge_rules){
    printf("\n\n-------------Wriring Merge Rules----------------\n");
    printf("\nWriting Merge Rules to file\n");
    FILE *fptr = fopen(filename, "w");  // fresh file
    fclose(fptr);
    fptr = fopen(filename, "a");
    for(size_t i = 0; i < merge_rules->len; i++){
        fprintf(fptr, "%c%s%c%c%s%c\n", 
            STX, 
            merge_rules->byte_pairs[i]->token1->val, 
            ETX,
            STX,
            merge_rules->byte_pairs[i]->token2->val,
            ETX
        );
    }
    fclose(fptr);
     printf("\n----------Done Writing Merge Rules--------------\n");
}

void tokenizer_read_merge_rules(char *filename, MergeRules *merge_rules){
    printf("\n\n----------Reading Merge Rules---------------\n");
    char *data = read_data_from_file(filename);
    size_t data_size = strlen(data);

    char token_text[48] = {0};
    size_t token_text_index = 0;
    size_t no_of_tokens_created = 0;
    Token *token1 = NULL;
    Token *token2 = NULL;
    bool read = false;
    for(size_t i = 0; i < data_size; i++){
        if(data[i] == STX){
            read = true;
            memset(token_text, 0, strlen(token_text));
            token_text_index = 0;
        }
        else if(data[i] == ETX){
            read = false;
            if(no_of_tokens_created == 0) token1 = tokenizer_create_token(token_text, token_text_index);
            else if(no_of_tokens_created == 1) token2 = tokenizer_create_token(token_text, token_text_index);
            no_of_tokens_created++;
            if(no_of_tokens_created == 2){
                BytePair *merge_rule = tokenizer_create_byte_pair(token1, token2, 1);
                merge_rules->byte_pairs[merge_rules->len++] = merge_rule;
                no_of_tokens_created = 0;
            }
        }
        if(read){
            token_text[token_text_index++] = data[i];
        }   
    }
    printf("--------Done Reading Merge Rules------------\n");
}

void tokenizer_train(Data *data, Vocab *vocab, MergeRules *merge_rules){
    printf("\n ========================================= Started Training ========================================= \n");
    BytePair *most_frequent_byte_pair = NULL;
    size_t itr = 0;
    while(1){
        printf("\n\n\n========================================= ITERATION %zu ========================================= \n", itr);
        BytePair **byte_pairs = calloc(MAX_BYTE_PAIRS, sizeof(BytePair*));
        size_t no_of_byte_pairs = tokenizer_get_byte_pairs(data, byte_pairs);
        
        if((itr+1) % 1000 == 0) {
            tokenizer_write_vocab("./vocab.txt", vocab);
            tokenizer_write_merge_rules("merge_rules.txt", merge_rules);
            tokenizer_print_byte_pairs(byte_pairs, no_of_byte_pairs);
        }
        printf("Max No of Byte Pairs: %zu\n", no_of_byte_pairs);
        if(no_of_byte_pairs == 0) break;
        most_frequent_byte_pair = tokenizer_get_most_frequent_byte_pair(byte_pairs, no_of_byte_pairs);
        Token *new_vocab_token = tokenizer_concat_tokens(most_frequent_byte_pair->token1, most_frequent_byte_pair->token2);
        vocab->tokens[vocab->len++] = new_vocab_token;
        merge_rules->byte_pairs[merge_rules->len++] = tokenizer_create_byte_pair(most_frequent_byte_pair->token1, most_frequent_byte_pair->token2, most_frequent_byte_pair->freq);
        tokenizer_merge(data, most_frequent_byte_pair);
        tokenizer_free_byte_pairs(byte_pairs, no_of_byte_pairs);
        itr++;
        printf("=============================================================================================");
    }
    //Adding Assci Characters
    for(int i = 0; i < 126; i++){
        vocab->tokens[vocab->len++] = tokenizer_create_token((char[]){i, '\0'}, 1);
    }
    printf("========================================= Done Training ========================================= \n");
}


void tokenizer_encode(char *data, int *token_ids, size_t *token_ids_len, Vocab *vocab, MergeRules *merge_rules){
    //printf("\n--------Encoding------------\n");
    size_t data_size = strlen(data);
    Word **words = calloc(MAX_WORDS, sizeof(Word*));
    size_t total_words = 0;
    char curr_word[48] = {0};
    size_t curr_word_index = 0;
    for(size_t i = 0; i < data_size; i++){
        char c =  tolower((unsigned char)data[i]);
        curr_word[curr_word_index++] = c;
        if (c == ' '    || c == '\n'    || c == '\t'    || c == '!' ||
            c == ';'    || c == ','     || c == ':'     || c == '"' ||
            c == '\''   || c == '.'     || c == '-'     || c == '?' ){
            Token **tokens = calloc(curr_word_index, sizeof(Token*));
            for(size_t j = 0; j < curr_word_index; j++){
                Token *new_token = tokenizer_create_token(&curr_word[j], 1);
                tokens[j] = new_token;
            }
            Word *new_word = tokenizer_create_word(tokens, curr_word_index);
            words[total_words++] = new_word;
            curr_word_index = 0;
            memset(curr_word, 0, strlen(curr_word));
        }
    }
    Data *new_data = calloc(1, sizeof(Data));
    new_data->words = words;
    new_data->len = total_words;
    
    for(size_t i = 0; i < merge_rules->len; i++){
        //tokenizer_print_byte_pair(merge_rules->byte_pairs[i]);
        tokenizer_merge(new_data, merge_rules->byte_pairs[i]);
    }
    // while(1){
    //     size_t merge_rule_applied = 0;
    //     for(size_t i = 0; i < new_data->len; i++){
    //         Word *word = new_data->words[i];
    //         for(size_t j = 1; j < word->len; j++){
    //             Token *token1 = word->tokens[j-1];
    //             Token *token2 = word->tokens[j];
    //             int res = tokenizer_get_merge_rule_index(merge_rules, token1, token2);
    //             if(res != -1){
    //                 Token *new_token = tokenizer_concat_tokens(token1, token2);
    //                 tokenizer_free_token(token1);
    //                 tokenizer_free_token(token2);
    //                 word->tokens[j-1] = new_token;
    //                 for(size_t k = j+1; k < word->len; k++){
    //                     word->tokens[k-1] =  word->tokens[k];
    //                 }
    //                 word->len--;
    //                 merge_rule_applied = 1;
    //             }
    //         }
    //     }
    //     if(merge_rule_applied == 0) break;
    // }

    //ize_t token_ids_len = 0;
    // size_t token_ids[MAX_WORDS] = {-1};
    for(size_t i = 0; i < new_data->len; i++){
        Word *word = new_data->words[i];
        //print_word(word);
        for(size_t j = 0; j < word->len; j++){
            int token_id = tokenizer_get_token_index(vocab, word->tokens[j]);
            token_ids[(*token_ids_len)++] = token_id;
        }
    }
    tokenizer_free_data(new_data);
    //printf("--------Done Encoding-------\n");
}

void tokenizer_decode(int *token_ids, Vocab *vocab){
    printf("\nDecoding\n");
    size_t token_id_index = 0;
    for(size_t i = 0; i < 100; i++){
        int token_id = token_ids[token_id_index++];
        //printf("Token id: %d\n", token_id);
        if(token_id == -1) break;
        printf("%s", vocab->tokens[token_id]->val);
    }

}