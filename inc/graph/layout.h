// #ifndef __LAYOUT_H__
// #define __LAYOUT_H__

// #include "./graph.h"
// #include <stdio.h>


// typedef enum QKVStrategy{
//     QKV_LAYOUT_HEAD_MAJOR_COMBINED,
//     QKV_LAYOUT_TOKEN_MAJOR_COMBINED
// } QKVLayoutStrategy;

// typedef struct QKVLayout{
//     size_t q_offset;
//     size_t k_offset;
//     size_t v_offset;
//     size_t stride_head;
//     size_t stride_token;
// } QKVLayout;

// // typedef enum AttentionScratchStrategy{

// // } AttentionScratchStrategy;


// // typedef struct AttentionScratchLayout{
// //     size_t stride_kt;
// //     size_t stride_attn_scores;
// // } AttentionScratchLayout;


// typedef struct LayoutStrategy{
//     QKVLayoutStrategy qkv_strategy;

// } LayoutStrategy;


// void plan_layouts(Graph *graph, LayoutStrategy strategy);
// #endif