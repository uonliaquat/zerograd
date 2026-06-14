#include "../../inc/graph/sampler.h"
#include <math.h>


static inline size_t greedy_sampling(float *logits, size_t vocab_size){
    float best_token_val = -INFINITY;
    size_t best_token_id = 0;
    for(size_t token_id = 0; token_id < vocab_size; token_id++){
        if(logits[token_id] > best_token_val){
            best_token_val = logits[token_id];
            best_token_id = token_id;
        }
    }
    return best_token_id;
}

size_t sample_token(SamplingStrategy strategy, float *logits, size_t vocab_size){
    switch(strategy){
        case SAMPLING_GREEDY: return greedy_sampling(logits, vocab_size);
        default: return greedy_sampling(logits, vocab_size);;
    }
}