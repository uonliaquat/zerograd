#ifndef __UTILS_H__
#define __UTILS_H__



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/utils.h"


static inline float rand_uniform(const float min, const float max){
    return min + (max - min) * ((float)rand() / RAND_MAX);
}


static inline void print_centered_heading(const char *heading) {
    //#define DEBUG
    #ifdef DEBUG
    const int width = 68;  // inner width between the bars
    int len = strlen(heading);

    int left_pad  = (width - len) / 2;
    int right_pad = width - len - left_pad;

    printf("\n");
    printf("+====================================================================+\n");
    printf("|%*s%s%*s|\n", left_pad, "", heading, right_pad, "");
    printf("+====================================================================+\n");
    #endif
}

static inline void create_filename(const char *base_path, const char *name, char *filename){
    memset(filename, 0, strlen(filename));
    memcpy(filename, base_path, strlen(base_path));
    strcat(filename, name);
}

static inline char * read_file(const char *filename){
    FILE * fptr = fopen(filename, "r");
    if(!fptr){
        perror("Error opening file");
        exit(-1);
    }
    fseek(fptr, 0, SEEK_END);
    long file_size = ftell(fptr);
    rewind(fptr);


    char *data = calloc(file_size, 1);
    if(!data){
        perror("data allocation failed\n");
        exit(-1);
    }
    size_t read = fread(data, 1, file_size, fptr);
    printf("Read %zu bytes from file\n", read);
    printf("file_size:: %ld\n\n", file_size);
    fclose(fptr);
    return data;
    
}

#endif