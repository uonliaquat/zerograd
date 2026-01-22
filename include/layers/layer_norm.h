#ifndef __LAYER_NORM_H__
#define __LAYER_NORM_H__


typedef struct LayerNorm{
    float std;
    float mean;
} LayerNorm;

LayerNorm layer_norm_init(size_t embed_dim);
LayerNorm layer_norm_forward();

#endif