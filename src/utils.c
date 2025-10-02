
#include <stdlib.h>
#include "../include/utils.h"


double rand_double(const double min, const double max){
    return min + (max - min) * ((double)rand() / RAND_MAX);
}