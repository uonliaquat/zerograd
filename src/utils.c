#include "../include/utils.h"

float generate_rand_val(float lower, float upper){
    return lower + ((float)rand() / RAND_MAX) * (upper - lower);
}