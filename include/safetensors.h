#ifndef __SAFETENSORS_H__
#define __SAFETENSORS_H__

#include "./tensor.h"
#include "./models/gpt.h"

void safetensors_load_model(const char *filename, GPTParams *params);
void safetensors_save_model(const char *filename, Tensor **tensors, size_t no_of_tensors);
//Tensor  safetensors_create_tensor(char *data, char *t_name);
// void    model_loader_print_params(Params *params);

#endif