#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_UNIQUE_WORDS 64000
#define MAX_BYTE_PAIRS 16000
#define MAX_MERGE_RULES 8000
#define MAX_VOCAB_SIZE 8000




struct Token{
    char *val;
    size_t len;
};

struct Word{
    struct Token **tokens;
    size_t len;
    size_t freq;
};

struct Corpus{
    struct Word **words;
    size_t len;
};

struct BytePair{
    struct Token *token1;
    struct Token *token2;
    size_t freq;
};

// struct BytePairs{
//     struct BytePair **pairs;
//     size_t len;
// };




struct Token *create_token(char *data, size_t data_size){
    struct Token *new_token = calloc(1, sizeof(struct Token));
    new_token->len = data_size;
    new_token->val = calloc(data_size, sizeof(char));
    memcpy(new_token->val, data, data_size);
    return new_token;
}


void free_token(struct Token *token){
    // printf("Freeing token: ");
    // printf("token len: %zu\n", token->len);
    if(token->len > 0){
        free(token->val);
        token->len = 0;
    }
}

void print_token(struct Token *token){
    printf("[%s] ", token->val);
}

struct Word *create_word(struct Token **tokens, size_t len){
    struct Word *new_word = calloc(1, sizeof(struct Word));
    new_word->len = len;
    new_word->tokens = tokens;
    new_word->freq = 1;
    return new_word;
}

void free_word(struct Word *word){
    for(size_t i = 0; i < word->len; i++){
        free_token(word->tokens[i]);
    }
    free(word);
    word->len = 0;
    word->freq = 0;
}

void print_word(struct Word *word){
    for(size_t i = 0; i < word->len; i++){
        print_token(word->tokens[i]);
    }
    printf(" | len: %zu | freq: %zu\n", word->len, word->freq);
}

char *read_data_from_file(char *filename){
    FILE *fptr = fopen(filename, "r");
    if(fptr == NULL){
        printf("Error opening file!");
        exit(1);
    }
    
    fseek(fptr, 0, SEEK_END);
    size_t data_size = ftell(fptr);
    rewind(fptr);

    char *data = malloc(data_size + 1);
    fread(data, 1, data_size, fptr);
    data[data_size] = '\0';

    fclose(fptr);
    return data;


}


size_t match_tokens(struct Token *token1, struct Token *token2){
    if(token1->len != token2->len) return 0;
    int res = strcmp(token1->val, token2->val);
    if (res == 0) return 1;
    return 0;
}

size_t match_words(struct Word *word1, struct Word *word2){
    if(word1->len != word2->len) return 0;
    for(size_t i = 0; i < word1->len; i++){
        size_t res = match_tokens(word1->tokens[i], word2->tokens[i]);
        if(res == 0) return res;
    }
    return 1;
}

int get_word_index(struct Word **words, size_t no_of_words, struct Word *word){
    for(size_t i = 0; i < no_of_words; i++){
        int res = match_words(words[i], word);
        if(res == 1) return i;
    }
    return -1;
}

struct Corpus *create_corpus(char *data){
    size_t data_size = strlen(data);

    struct Word **words = calloc(MAX_UNIQUE_WORDS, sizeof(struct Word*));
    size_t total_unique_words = 0;
    char curr_word[48] = {0};
    size_t curr_word_index = 0;
    for(size_t i = 0; i < data_size; i++){
        char c =  tolower((unsigned char)data[i]);
        curr_word[curr_word_index++] = c;
        if (c == ' '    || c == '\n'    || c == '\t'    || c == '!' ||
            c == ';'    || c == ','     || c == ':'     || c == '"' ||
            c == '\''   || c == '.'     || c == '-'     || c == '?' ){
            
            struct Token **tokens = calloc(curr_word_index, sizeof(struct Token*));
            for(size_t j = 0; j < curr_word_index; j++){
                struct Token *new_token = create_token(&curr_word[j], 1);
                tokens[j] = new_token;
            }
            struct Word *new_word = create_word(tokens, curr_word_index);
            int word_index = get_word_index(words, total_unique_words, new_word);
            if(word_index == -1){
                words[total_unique_words++] = new_word;
            }
            else{
                words[word_index]->freq++;
            }
            curr_word_index = 0;
            memset(curr_word, 0, strlen(curr_word));
    }
    }
    struct Corpus *new_corpus = calloc(1, sizeof(struct Corpus));
    new_corpus->words = words;
    new_corpus->len = total_unique_words;
    return new_corpus;
}



void free_corpus(struct Corpus *corpus){
    for(size_t i = 0; i < corpus->len; i++){
        free_word(corpus->words[i]);
    }
    corpus->len = 0;
}

void print_corpus(struct Corpus *corpus){
    printf("No of Unique Words in Corpus: %zu\n", corpus->len);
    for(size_t i = 0; i < corpus->len; i++){
        print_word(corpus->words[i]);
    }
}


struct BytePair *create_byte_pair(struct Token *token1, struct Token *token2, size_t freq){
    struct BytePair *new_byte_pair = calloc(1, sizeof(struct BytePair));
    new_byte_pair->token1 = create_token(token1->val, token1->len);
    new_byte_pair->token2 = create_token(token2->val, token2->len);
    new_byte_pair->freq = freq;
    return new_byte_pair;
}

void free_byte_pair(struct BytePair *byte_pair){
    free_token(byte_pair->token1);
    free_token(byte_pair->token2);
    byte_pair->freq = 0;
}

void print_byte_pair(struct BytePair *byte_pair){
    print_token(byte_pair->token1);
    print_token(byte_pair->token2);
    printf(" | freq: %zu\n", byte_pair->freq);
}

void print_byte_pairs(struct BytePair **byte_pairs, size_t no_of_byte_pairs){
    printf("Byte Pairs:\n");
    for(size_t i = 0; i < no_of_byte_pairs; i++){
        print_byte_pair(byte_pairs[i]);
    }
}

int get_byte_pair_index(struct BytePair **pairs, size_t no_of_byte_pairs, struct BytePair *byte_pair){
    for(size_t i = 0; i < no_of_byte_pairs; i++){
       int res1 = match_tokens(pairs[i]->token1, byte_pair->token1);
       int res2 = match_tokens(pairs[i]->token2, byte_pair->token2);
       if(res1 == 1 && res2 == 1) return i;
    }
    return -1;
}

size_t get_byte_pairs(struct Corpus *corpus, struct BytePair **byte_pairs){
    size_t no_of_byte_pairs = 0;
    for(size_t i = 0; i < corpus->len; i++){
        struct Word *word = corpus->words[i];
        for(size_t j = 1; j < word->len; j++){
            struct Token *token_prev = word->tokens[j-1];
            struct Token *token_curr = word->tokens[j];
            struct BytePair *byte_pair = create_byte_pair(token_prev, token_curr, 1);
            size_t byte_pair_index = get_byte_pair_index(byte_pairs, no_of_byte_pairs, byte_pair);
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

void free_byte_pairs(struct BytePair **byte_pairs, size_t no_of_byte_pairs){
    for(size_t i = 0; i < no_of_byte_pairs; i++){
        free_byte_pair(byte_pairs[i]);
    }
    no_of_byte_pairs = 0;
}

struct Token *concat_tokens(struct Token *token1, struct Token *token2){
    size_t new_token_len = token1->len + token2->len;
    struct Token *new_token = calloc(1, sizeof(struct Token));
    new_token->len = new_token_len;
    new_token->val = calloc(new_token_len, sizeof(char));
    memcpy(new_token->val, token1->val, token1->len);
    memcpy(new_token->val + token1->len, token2->val, token2->len);
    return new_token;
}


struct BytePair *get_most_frequent_byte_pair(struct BytePair **byte_pairs, size_t no_of_byte_pairs){
    if(no_of_byte_pairs == 0){
        printf("\nno_of_byte_pairs: %zu\n", no_of_byte_pairs);
        return NULL;
    } 
    struct BytePair *most_frequent_byte_pair = create_byte_pair(byte_pairs[0]->token1, byte_pairs[0]->token2, byte_pairs[0]->freq);
    for(size_t i = 1; i <no_of_byte_pairs; i++){
        if(byte_pairs[i]->freq > most_frequent_byte_pair->freq){
            free_byte_pair(most_frequent_byte_pair);
            most_frequent_byte_pair = create_byte_pair(byte_pairs[i]->token1, byte_pairs[i]->token2, byte_pairs[i]->freq);
        }
    }
    return most_frequent_byte_pair;
}

void merge(struct Corpus *corpus, struct BytePair *most_frequent_byte_pair){
    for(size_t i = 0; i < corpus->len; i++){
        struct Word *word = corpus->words[i];
        for(size_t j = 1; j < word->len; j++){
            struct Token *token1 = word->tokens[j-1];
            struct Token *token2 = word->tokens[j];
            int res1 = match_tokens(token1, most_frequent_byte_pair->token1);
            int res2 = match_tokens(token2, most_frequent_byte_pair->token2);
            if(res1 == 1 && res2 == 1){
                // printf("\nWord before merge: "); print_word(word);
                // printf("  token1: "); print_token(token1);
                // printf("  token2: "); print_token(token2);
                struct Token *new_token = concat_tokens(token1, token2);
                free_token(token1);
                free_token(token2);
                word->tokens[j-1] = new_token;
                for(size_t k = j+1; k < word->len; k++){
                    // printf("Token k-1 "); print_token(word->tokens[k-1]);
                    // printf(" | Token k"); print_token(word->tokens[k]);
                    
                    word->tokens[k-1] =  word->tokens[k];
                }
                word->len--;
                // printf("\nWord after merge: "); print_word(word);
                // printf("  new_token: "); print_token(new_token);
            }
        }
    }
}

void write_vocab(char *filename, struct Token **vocab, size_t vocab_size){
    printf("\nWriting Vocab to file");
    FILE *fptr = fopen(filename, "a");
    for(size_t i = 0; i < vocab_size; i++){
        fprintf(fptr, "{%zu}{%s}\n", i, vocab[i]->val);
    }
    fclose(fptr);
}
void write_merge_rules(char *file_name, struct BytePair **merge_rules, size_t vocab_size){
    printf("\nWriting Merge Rules to file\n");
    FILE *fptr = fopen(file_name, "a");
    for(size_t i = 0; i < vocab_size; i++){
        fprintf(fptr, "{%s}{%s}\n", merge_rules[i]->token1->val, merge_rules[i]->token2->val);
    }
       fclose(fptr);
}

size_t train_bpe(struct Corpus *corpus, struct Token **vocab, struct BytePair **merge_rules){
    struct BytePair *most_frequent_byte_pair = NULL;
    size_t itr = 0;
    while(1){
        printf("\n\n\n========================================= ITERATION %zu ========================================= \n", itr);
        struct BytePair **byte_pairs = calloc(MAX_BYTE_PAIRS, sizeof(struct BytePair*));
        size_t no_of_byte_pairs = get_byte_pairs(corpus, byte_pairs);
        //print_byte_pairs(byte_pairs, no_of_byte_pairs);
        printf("no_of_byte_pairs %zu\n", no_of_byte_pairs);

        most_frequent_byte_pair = get_most_frequent_byte_pair(byte_pairs, no_of_byte_pairs);
        //printf("\nMost frquent byte pair: "); print_byte_pair(most_frequent_byte_pair);
        if(most_frequent_byte_pair == NULL) break;
        struct Token *new_vocab_token = concat_tokens(most_frequent_byte_pair->token1, most_frequent_byte_pair->token2);
        vocab[itr] = new_vocab_token;
        merge_rules[itr] = create_byte_pair(most_frequent_byte_pair->token1, most_frequent_byte_pair->token2, most_frequent_byte_pair->freq);
        merge(corpus, most_frequent_byte_pair);
        //print_corpus(corpus);
        free_byte_pairs(byte_pairs, no_of_byte_pairs);
        itr++;
        printf("=============================================================================================");
    }
    //free_byte_pair(most_frequent_byte_pair);
    return itr;

}


int main(){
    char *data = read_data_from_file("/Users/uonliaquat/workspace/zerograd/tokenizer/the-verdict.txt");
    //char data[] = "hug, pug, pun, bun, hugs pug hug hug bun fun";
    // char data[] =  "the fox jumps over the lazy dog. "
    //         "the fox jumps high over the lazy hound. "
    //         "foxes and hounds play in the forest. "
    //         "the quick brown fox jumps over the lazy dog again. "
    //         "foxes jump quickly, dogs bark loudly, and the forest echoes. "
    //         "the lazy dog sleeps while the fox jumps. "
    //         "hounds chase foxes, foxes escape, foxes jump, hounds bark, dogs run. "
    //         "the fox and the dog meet again in the forest. "
    //         "dogs and foxes interact, jump, play, and run in the woods. "
    //         "the quick brown fox jumps over lazy dogs repeatedly. "
    //         "jumping foxes over lazy dogs is fun. "
    //         "brown foxes jump over the quick hounds repeatedly. "
    //         "lazy dogs bark while quick foxes jump over them. "
    //         "the forest echoes with jumps, barks, and foxes running. "
    //         "foxes, hounds, dogs, and the quick brown fox interact together.";

    printf("Data:\n%s\n\n", data);
    printf("data_size: %zu\n", strlen(data));
    struct Corpus *corpus = create_corpus(data);
    print_corpus(corpus);

    struct Token **vocab = calloc(MAX_VOCAB_SIZE, sizeof(struct Token*));
    struct BytePair **merge_rules = calloc(MAX_MERGE_RULES, sizeof(struct BytePiar*));
    size_t vocab_size = train_bpe(corpus, vocab, merge_rules);

    write_vocab("./vocab.txt", vocab, vocab_size);
    write_merge_rules("merge_rules.txt", merge_rules, vocab_size);

    //Free Vocab and merge_rules
    for(size_t i = 0; i < vocab_size; i++){
        free_token(vocab[i]);
        free_byte_pair(merge_rules[i]);
    }
    free_corpus(corpus);
    
}