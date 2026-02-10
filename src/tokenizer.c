#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

#include "../include/tokenizer.h"
#include "../include/utils.h"

static Corpus corpus;
static BytePairs byte_pairs;
static Vocab vocab;
static BytePairs merge_rules;


// int tokenizer_cmp_tokens(Token *token1, Token *token2){
//     for(size_t i = 0; i < MAX_TOKEN_LEN; i++){
//         int res = strcmp(token1->token, token2->token);
//         if(res != 0) return res;
//     }
//     return 0;
// }

static inline int tokenizer_get_word_index(Word *word_to_find){
    for(size_t i = 0; i < corpus.len; i++){
        Word word = corpus.words[i];
        int matched = 1;
        for(size_t j = 0; j < MAX_WORD_LEN; j++){
            if(word.token_ids[j] != word_to_find->token_ids[j]){
                matched = 0;
                break;
            }
        }
        if(matched == 1) return i;
    }
    return -1;
}

static inline void tokenizer_copy_word(Word *dest, Word *src){
    memcpy(dest->token_ids, src->token_ids, MAX_WORD_LEN * sizeof(int16_t));
    dest->freq = src->freq;
}
static inline Word tokenizer_init_word(int16_t *token_ids){
    Word new_word;
    memcpy(new_word.token_ids, token_ids, MAX_WORD_LEN*sizeof(int16_t));
    new_word.freq = 1;
    return new_word;
}

static inline void tokenizer_clear_word(Word *word){
    memset(word->token_ids, -1, MAX_WORD_LEN*sizeof(int16_t));
}


static void tokenizer_print_word(Word *word){
    size_t k = 0;
    while(k < MAX_WORD_LEN && word->token_ids[k] != -1){
        printf("[%s]", vocab.tokens[word->token_ids[k]].token);
        k++;
    }
    printf("\n");
}

static inline void tokenizer_copy_byte_pair(BytePair *dest, BytePair *src){
    dest->token1_id = src->token1_id;
    dest->token2_id = src->token2_id;
    dest->freq = src->freq;
}

static inline void tokenizer_init_byte_pair(BytePair *byte_pair, int16_t token1_id, int16_t token2_id, int32_t freq){
    byte_pair->token1_id = token1_id;
    byte_pair->token2_id = token2_id;
    byte_pair->freq = freq;
}

static inline int tokenizer_get_byte_pair_index(BytePair *byte_pair_to_find){
    for(size_t i = 0; i < byte_pairs.len; i++){
        if(byte_pairs.pairs[i].token1_id == byte_pair_to_find->token1_id && 
           byte_pairs.pairs[i].token2_id == byte_pair_to_find->token2_id ){
            return i;
        }
    }
    return -1;
}

static inline void tokenizer_print_vocab(){
    printf("Vocab Len: %d\n", vocab.len);
    for(size_t i = 0; i < vocab.len; i++){
        printf("Token ID: %zu | Token: %s\n", i, vocab.tokens[i].token);
    }
}

static inline void tokenizer_init_corpus(char *filename){
    printf("\n*************************** tokenizer_init_corpus (START) ***************************\n");
    FILE *file_ptr = fopen(filename, "r");
    if(file_ptr == NULL){
        printf("Couldn't open file %s\n", filename);
        exit(1);
    }
    for(size_t i = 0; i < 256; i++){
        char c = i;
        memset(vocab.tokens[vocab.len].token, '\0', MAX_TOKEN_LEN);
        memcpy(vocab.tokens[vocab.len].token, &c, 1);
        vocab.len++;
    }
    // for(size_t i = 0; i < vocab.len; i++){
    //     printf("Token ID: %zu | Token: %s\n", i, vocab.tokens[i].token);
    // }
    //tokenizer_print_vocab();
    corpus.len = 0;
    int ch;
    int16_t curr_word_token_ids[4096];
    memset(curr_word_token_ids, -1, 4096*sizeof(int16_t));
    int16_t curr_word_next_token_index = 0;
    while((ch = fgetc(file_ptr)) != EOF){
        char c = tolower((unsigned char) ch);
        curr_word_token_ids[curr_word_next_token_index++] = c;
        if (c == ' '    || c == '\n'    || c == '\t'    || c == '!' ||
            c == ';'    || c == ','     || c == ':'     || c == '"' ||
            c == '\''   || c == '.'     || c == '-'     || c == '?' ){
            if(curr_word_next_token_index > MAX_TOKEN_LEN){
                memset(curr_word_token_ids, -1, 4096*sizeof(int16_t));
                curr_word_next_token_index = 0;
                break;
            }
            Word new_word = tokenizer_init_word(curr_word_token_ids);
            int word_index = tokenizer_get_word_index(&new_word);
            if(word_index == -1){
                tokenizer_copy_word(&corpus.words[corpus.len], &new_word);
                corpus.len++;
                //printf("No of Unique Words %d\n", corpus.len);
            }
            else{
                corpus.words[word_index].freq++;
            }
            memset(curr_word_token_ids, -1, 4096*sizeof(int16_t));
            curr_word_next_token_index = 0;
        }
    }
    printf("**************************** tokenizer_init_corpus (END) ****************************\n");
}

static inline BytePair tokenizer_get_most_frequent_byte_pair(){
    printf("******************* tokenizer_get_most_frequent_byte_pair (START) *******************\n");
    BytePair byte_pair;
    BytePair most_frequent_byte_pair;
    most_frequent_byte_pair.freq = 0;
    byte_pairs.len = 0;
    for(size_t i = 0; i < corpus.len; i++){
        Word *word = &corpus.words[i];
        size_t j = 0;
        int16_t token_id_prev, token_id_curr;
        while(j < MAX_WORD_LEN && (token_id_prev = word->token_ids[j]) != -1 && (token_id_curr = word->token_ids[++j]) != -1){
            tokenizer_init_byte_pair(&byte_pair, token_id_prev, token_id_curr, 1);
            //printf("token1: %s, token2: %s, freq: %d \n", vocab[byte_pair.token1_id].token, vocab[byte_pair.token2_id].token, byte_pair.freq);
            int byte_pair_index = tokenizer_get_byte_pair_index(&byte_pair);
             if(byte_pair_index == -1){
                //Create new byte_pair
                byte_pair_index = byte_pairs.len;
                tokenizer_copy_byte_pair(&byte_pairs.pairs[byte_pair_index], &byte_pair);
                byte_pairs.len++;
            }
            else{
                byte_pairs.pairs[byte_pair_index].freq += word->freq;
            }
            if(most_frequent_byte_pair.freq <  byte_pairs.pairs[byte_pair_index].freq){
                tokenizer_copy_byte_pair(&most_frequent_byte_pair, &byte_pairs.pairs[byte_pair_index]);
            }

        }
    }
    printf("most_frequent_byte_pair freq: %d\n", most_frequent_byte_pair.freq);
    printf("******************** tokenizer_get_most_frequent_byte_pair (END) ********************\n");
    return most_frequent_byte_pair;
}

static inline void tokenizer_add_token_to_vocab(BytePair *byte_pair){
    int token1_len = strlen(vocab.tokens[byte_pair->token1_id].token);
    int token2_len = strlen(vocab.tokens[byte_pair->token2_id].token);
    // Token new_token;
    // memcpy(new_token.token, token1, strlen(token1));
    // memcpy(new_token.token+strlen(token1), token2, strlen(token2));

    memcpy(vocab.tokens[vocab.len].token, vocab.tokens[byte_pair->token1_id].token, token1_len);
    memcpy(vocab.tokens[vocab.len].token + token1_len, vocab.tokens[byte_pair->token2_id].token, token2_len);
    vocab.len++;
    printf("******************************** Token Added to Vocab *******************************\n");
    printf("token: %s\n", vocab.tokens[vocab.len-1].token);
}

static inline void tokenizer_add_merge_rule_to_merge_rules(BytePair *byte_pair){
    merge_rules.pairs[merge_rules.len].token1_id = byte_pair->token1_id;
    merge_rules.pairs[merge_rules.len].token2_id = byte_pair->token2_id;
    merge_rules.pairs[merge_rules.len].freq = byte_pair->freq;
    merge_rules.len++;
    printf("****************************** Merge Rule Added to Vocab ****************************\n");
}


static inline void tokenizer_merge(BytePair *byte_pair){
    printf("****************************** tokenizer_merge (START) ******************************\n");
    for(size_t i = 0; i < corpus.len; i++){
        Word *word = &corpus.words[i];

        int16_t updated_token_ids[MAX_WORD_LEN];
        memset(updated_token_ids, -1, MAX_WORD_LEN*sizeof(int16_t));
        
        int16_t updated_token_id_index = 0;

        size_t j = 0;
        int16_t token_id_curr, token_id_next, new_token_id;
        int word_updated = 0;
        while(j < MAX_WORD_LEN-1 && (token_id_curr = word->token_ids[j]) != -1 && (token_id_next = word->token_ids[j+1]) != -1){
            if(token_id_curr == byte_pair->token1_id && token_id_next == byte_pair->token2_id){
                new_token_id = vocab.len-1;
                j++;
                word_updated = 1;
            }
            else{
                new_token_id = token_id_curr;
            }
            updated_token_ids[updated_token_id_index++] = new_token_id;
            j++;
        }
        // if(word_updated == 1) {
        //     printf("Before: "); 
        //     tokenizer_print_word(word);
        // }
        updated_token_ids[updated_token_id_index] = word->token_ids[j];
        memset(word->token_ids, -1, MAX_WORD_LEN*sizeof(int16_t));
        memcpy(word->token_ids, updated_token_ids, MAX_WORD_LEN*sizeof(int16_t));
        // if(word_updated == 1){
        //     printf("After:  "); 
        //     tokenizer_print_word(word);
        // }

    }
    printf("******************************* tokenizer_merge (END) *******************************\n"); 

}

static inline void tokenizer_write_vocab(char *filename){
    printf("\n************************************** Writing Vocab **************************************\n");
    FILE *fptr = fopen(filename, "w");  // fresh file
    fclose(fptr);
    fptr = fopen(filename, "a");
    for(size_t i = 0; i < vocab.len; i++){
        fprintf(fptr, "%c%s%c\n", 
            STX, 
            vocab.tokens[i].token,
            ETX
        );
    }
    fclose(fptr);
    printf("*********************************** Done Writing Vocab ************************************\n");
}


void tokenizer_read_vocab(const char *filename, Vocab *vocab){
    printf("\n************************************** Reading Vocab **************************************\n");
    FILE *fptr = fopen(filename, "r");  // fresh file
    if(fptr == NULL){
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    vocab->len = 0;
    size_t token_index = 0;
    int c;  // must be int to handle EOF
    while ((c = fgetc(fptr)) != EOF) {
        vocab->tokens[vocab->len].token[token_index++] = c;
        if(token_index == 64){
            vocab->len++;
            token_index = 0;
        } 
    }
    fclose(fptr);
    printf("*********************************** Done Reading Vocab ************************************\n");
}

void tokenizer_write_merge_rules(char *filename){
    printf("\n************************************ Wriring Merge Rules **********************************\n");
    FILE *fptr = fopen(filename, "w");  // fresh file
    fclose(fptr);
    fptr = fopen(filename, "a");
    for(size_t i = 0; i < merge_rules.len; i++){
        fprintf(fptr, "%c%d%c%c%d%c\n", 
            STX, 
            merge_rules.pairs[i].token1_id, 
            ETX,
            STX,
            merge_rules.pairs[i].token2_id,
            ETX
        );
    }
    fclose(fptr);
    printf("********************************** Done Writing Merge Rules *******************************\n");
}

void tokenizer_train(char *filename){
    tokenizer_init_corpus(filename);
    size_t itr = 0;
    while(1){
        printf("************************************* itr (%zu) *************************************\n", itr);
        BytePair most_frequent_byte_pair = tokenizer_get_most_frequent_byte_pair();
        if(most_frequent_byte_pair.freq == 0 || vocab.len == MAX_VOCAB_SIZE) break;
        tokenizer_add_merge_rule_to_merge_rules(&most_frequent_byte_pair);
        tokenizer_add_token_to_vocab(&most_frequent_byte_pair);
        tokenizer_merge(&most_frequent_byte_pair);
        if(itr % 1000 == 0){
            tokenizer_print_vocab();
            tokenizer_write_vocab("./vocab.txt");
            tokenizer_write_merge_rules("./merge_rules.txt");
        }
        itr++;
    }

}

// int main(){
//     // /Users/uonliaquat/workspace/zerograd/the-verdict.txt
//     tokenizer_train("/Users/uonliaquat/workspace/zerograd/dataset/shakespeare.txt");


// }