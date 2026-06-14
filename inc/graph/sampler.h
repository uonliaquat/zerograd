#ifndef __SAMPLER_H__
#define __SAMPLER_H__

#include <stdio.h>

typedef enum SamplingStrategy{
    SAMPLING_GREEDY
} SamplingStrategy;

size_t sample_token(SamplingStrategy strategy, float *embed, size_t n);

#endif