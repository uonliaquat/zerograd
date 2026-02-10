
#include "../../../include/models/gpt2/multi_layer_perceptron.h"
#include "../../../include/tensor.h"
#include <string.h>
#include <stdlib.h>

static inline void mlp_workspace_init(MLPWorkspace *workspace, char *name){

    char tmp_name[256] = "\0";
    snprintf(tmp_name, sizeof(tmp_name), "%s.gelu", name);
    tensor_reset(&workspace->gelu, tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s", name);
    tensor_reset(&workspace->output, tmp_name);
}

static inline void mlp_workspace_free(MLPWorkspace *workspace){
    tensor_free(&workspace->gelu);
    tensor_free(&workspace->output);
}

MultiLayerPerceptron multi_layer_perceptron_init(MultiLayerPerceptronParams *params, char *name){
    MultiLayerPerceptron mlp;

    strcpy(mlp.name, name);

    char tmp_name[256] = "\0";
    snprintf(tmp_name, sizeof(tmp_name), "%s.c_fc", name);
    mlp.c_fc    = linear_layer_init(&params->c_fc,  tmp_name);

    snprintf(tmp_name, sizeof(tmp_name), "%s.c_proj", name);
    mlp.c_proj  = linear_layer_init(&params->c_proj, tmp_name);

    mlp_workspace_init(&mlp.workspace, name);
    return mlp;
}

void multi_layer_perceptron_free(MultiLayerPerceptron *mlp){
    mlp_workspace_free(&mlp->workspace);
    linear_layer_free(&mlp->c_fc);
    linear_layer_free(&mlp->c_proj);
}


void multi_layer_perceptron_forward(MultiLayerPerceptron *mlp, Tensor *x){
    //print_centered_heading("Feed Forward Network");
    linear_layer_forward(&mlp->c_fc,     x);
    tensor_gelu_(&mlp->c_fc.workspace.output, &mlp->workspace.gelu);
    linear_layer_forward(&mlp->c_proj,  &mlp->workspace.gelu);
    tensor_copy_(&mlp->c_proj.workspace.output, &mlp->workspace.output);
}

void multi_layer_perceptron_write(MultiLayerPerceptron *mlp, Tensor **tensors, size_t *tensors_len){
    linear_layer_write(&mlp->c_fc, tensors, tensors_len);
    tensors[(*tensors_len)++] = &mlp->workspace.gelu;
    linear_layer_write(&mlp->c_proj, tensors, tensors_len);
    tensors[(*tensors_len)++] = &mlp->workspace.output;
}
