#ifndef __SAFETENSORS_H__
#define __SAFETENSORS_H__

#include "./tensor.h"
#include "./models/gpt.h"

void safetensors_load_model(const char *filename, GPTParameters *gpt_params);
void safetensors_save_model(const char *filename, GPTParameters *gpt_params);
//Tensor  safetensors_create_tensor(char *data, char *t_name);
// void    model_loader_print_params(Params *params);

#endif