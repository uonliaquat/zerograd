
#include <stdlib.h>
#include <stdio.h>
#include "../include/utils.h"


double rand_double(const double min, const double max){
    return min + (max - min) * ((double)rand() / RAND_MAX + 1.0);
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
