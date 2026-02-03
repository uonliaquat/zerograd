
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/utils.h"


static inline double rand_uniform(const double min, const double max){
    return min + (max - min) * ((double)rand() / RAND_MAX);
}


static inline void print_centered_heading(const char *heading) {
    const int width = 68;  // inner width between the bars
    int len = strlen(heading);

    int left_pad  = (width - len) / 2;
    int right_pad = width - len - left_pad;

    printf("\n");
    printf("+====================================================================+\n");
    printf("|%*s%s%*s|\n", left_pad, "", heading, right_pad, "");
    printf("+====================================================================+\n");
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



// char *read_data_from_file(char *filename){
//     FILE *fptr = fopen(filename, "r");
//     if(fptr == NULL){
//         printf("Error opening file!");
//         exit(1);
//     }
    
//     fseek(fptr, 0, SEEK_END);
//     size_t data_size = ftell(fptr);
//     rewind(fptr);

//     char *data = malloc(data_size + 1);
//     fread(data, 1, data_size, fptr);
//     data[data_size] = '\0';

//     fclose(fptr);
//     return data;
// }
