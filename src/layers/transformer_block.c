
// #include "../../include/layers/transformer_block.h"
// #include "../../include/utils.h"
// #include "../../include/models/gpt.h"

// #include<string.h>

// // TransformerBlock transformer_block_init(size_t context_len, size_t embed_dim, size_t n_heads, size_t n_layers, size_t drop_rate, bool qkv_bias, bool decoder){
// //     TransformerBlock transformer_block;
// //     transformer_block.n_layers = n_layers;
// //     transformer_block.decoder = decoder;
// //     transformer_block.transformer_layers = calloc(n_layers, sizeof(TransformerLayer));
// //     for(size_t layer_no = 0; layer_no < n_layers; layer_no++){
// //         transformer_block.transformer_layers[layer_no] = transformer_layer_init(context_len, embed_dim, n_heads, qkv_bias, false);
// //     }
// //     return transformer_block;
// // }

// // TransformerBlock transformer_block_init(GPTConfig *gpt_config, GPTParameters *gpt_params){
// //     TransformerBlock transformer_block;
// //     transformer_block.n_layers = gpt_config->n_layers;
// //     transformer_block.decoder = true;
// //     transformer_block.transformer_layers = calloc(gpt_config->n_layers, sizeof(TransformerLayer));
// //     for(size_t layer_no = 0; layer_no < gpt_config->n_layers; layer_no++){
// //         transformer_block.transformer_layers[layer_no] = transformer_layer_init(gpt_config, gpt_params);
// //     }
// //     return transformer_block;
// // }

// TransformerBlock transformer_block_init(TransformerBlockParams *params, size_t context_len, size_t embed_dim, size_t n_heads, size_t n_layers, size_t drop_rate, bool qkv_bias, bool decoder){
//     TransformerBlock transformer_block;
//     transformer_block.params = params;
//     transformer_block.n_layers = n_layers;
//     transformer_block.decoder = decoder;
//     transformer_block.transformer_layers = calloc(n_layers, sizeof(TransformerLayer));
//     for(size_t layer_no = 0; layer_no < n_layers; layer_no++){
//         //transformer_block.transformer_layers[layer_no] = transformer_layer_init(gpt_config, gpt_params);
//     }
//     return transformer_block;
// }

// void transformer_block_free(TransformerBlock *transformer_block){
//     for(size_t layer_no = 0; layer_no < transformer_block->n_layers; layer_no++){
//         transformer_layer_free(&transformer_block->transformer_layers[layer_no]);
//     }
// }

// void transformer_block_forward(TransformerBlock *transformer_block, Tensor *input){
//     print_centered_heading("Transfromer Block");
//     for(size_t layer_no = 0; layer_no < transformer_block->n_layers; layer_no++){
//         char heading[512] = "\0";
//         snprintf(heading, 512, "%s %zu", "Transformer Layer", layer_no);
//         print_centered_heading(heading);
//         transformer_layer_forward(&transformer_block->transformer_layers[layer_no], input, transformer_block->decoder);
//     }
// }

// // void transformer_block_print(TransformerBlock *transformer_block, const char *heading){
// //     print_centered_heading(heading);
// //     for(size_t layer_no = 0; layer_no < transformer_block->n_layers; layer_no++){
// //         char heading[512] = "\0";
// //         snprintf(heading, 512, "%s_%zu", "transformer_layer_", layer_no);
// //         transformer_layer_print(&transformer_block->transformer_layers[layer_no], heading);
// //     }

// // }


// // void transformer_block_write(TransformerBlock *transformer_block, const char *base_path){
// //     for(size_t layer_no = 0; layer_no < transformer_block->n_layers; layer_no++){
// //         char filename[512] = "/0";
// //         snprintf(filename, 512, "%stransformer_layer_%zu", base_path, layer_no);
// //         transformer_layer_write(&transformer_block->transformer_layers[layer_no], filename);
// //     }
// // }