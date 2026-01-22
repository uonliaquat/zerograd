#ifndef __ENCODER_H__
#define __ENCODER_H__

#include "./layers/self_attention.h"



typedef struct FeedForward{
    Tensor input_layer;
    Tensor hidden_layer;
    Tensor output_layer;
} FeedForward;


typedef struct AddAndNorm{
    Tensor LayerNorm;

} AddAndNorm;

typedef struct TransformerEncoder{
    SelfAttentionLayer self_attention_layer;
    FeedForward feed_forward;
    
} TransformerEncoder;

#endif