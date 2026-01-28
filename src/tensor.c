#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "../include/tensor.h"
#include "../include/utils.h"

Tensor tensor_init(void *data, const size_t * shape, const size_t ndim, const DataType dtype, const bool requires_grad, const bool random_init){

    Tensor tensor = { 0 };


    tensor.dtype = dtype;

    tensor.ndim = ndim;
    tensor.elem_size = tensor_dtype_size(dtype);
    tensor.requires_grad = requires_grad;


    // Initializing shape
    memset(tensor.shape, 0, sizeof(tensor.shape));
    memcpy(tensor.shape, shape, sizeof(size_t)*ndim);


    // Initialzing size, (total no of elements in tensor)
    tensor.size = 1;
    for(size_t i = 0; i < ndim; i++)
        tensor.size = tensor.size * tensor.shape[i];
    
    
    // Initializing stride
    tensor.stride[0] = 1;
    tensor.stride[1] = 1;
    tensor.stride[2] = 1;
    //memset(tensor.stride, 0, sizeof(tensor.stride));
    //ndim = 3
    for(int i = ndim-2; i >= 0; i--){
        tensor.stride[i] = tensor.stride[i+1] * tensor.shape[i+1];
    }

    // tensor.stride[3] = 1;
    // for(size_t i = 3; i > 0; i--){
    //     tensor.stride[i-1] = tensor.shape[i];
    // }

    // Initializing data
    tensor.data = (void*)calloc(tensor.size,  tensor.elem_size);
    if(data != NULL)
        memcpy(tensor.data, data, tensor.elem_size * tensor.size);
    else if(random_init == true){
        for(size_t i = 0; i < tensor.size; i++){
            double rand_number = rand_uniform(-1.0, 1.0);
            ((double*)tensor.data)[i] = rand_number;
        }
    }
    else{
        for(size_t i = 0; i < tensor.size; i++){
            double elem = 0;
            ((double*)tensor.data)[i] = elem;
        }
    }
    return tensor;
}

void tensor_free(const Tensor *tensor){
    free(tensor->data);
}

Tensor tensor_copy(Tensor *input){
    Tensor output_tensor = tensor_init(
        NULL, 
        input->shape,
        input->ndim,
        input->dtype,
        input->requires_grad,
        true
    );
    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k});
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k}, elem);
            }
        }
    }
    return output_tensor;
}

double tensor_get_elem(const Tensor *tensor, size_t *coords){
    // size_t rows = tensor->shape[0];
    // size_t cols = tensor->shape[1];
    // assert(row < rows && col < cols);
    size_t index = 0;
    for(size_t d = 0; d < tensor->ndim; d++){
        index += coords[d] * tensor->stride[d];
    }
    return ((double*)tensor->data)[index];
}

void tensor_put_elem(Tensor *tensor, size_t *coords, double elem){
    size_t index = 0;
    for(size_t d = 0; d < tensor->ndim; d++){
        index += coords[d] * tensor->stride[d];
    }
    ((double*)tensor->data)[index] = elem;
}

Tensor tensor_transpose(const Tensor *tensor){
    Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){tensor->shape[1],  
        tensor->shape[0]}, 
        2,
        tensor->dtype,
        tensor->requires_grad,
        true
    );

    for(size_t i = 0; i < tensor->shape[0]; i++){
        for(size_t j = 0; j < tensor->shape[1]; j++){
            double elem = tensor_get_elem(tensor, (size_t[]){i, j});
            tensor_put_elem(&output_tensor, (size_t[]){j, i}, elem);
        }
    }
    return output_tensor;
}


Tensor tensor_softmax(Tensor *tensor, size_t dim){
    Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){tensor->shape[1],  
        tensor->shape[0]}, 
        2,
        tensor->dtype,
        tensor->requires_grad,
        true
    );
    if(dim == 1){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            double exp_sum = 0;
            for(size_t j = 0; j < tensor->shape[1]; j++){
                double elem = tensor_get_elem(tensor, (size_t[]){i, j});
                exp_sum = exp_sum + expf(elem);
            }
            for(size_t j = 0; j < tensor->shape[1]; j++){
                double elem = tensor_get_elem(tensor, (size_t[]){i, j});

                double new_elem = expf(elem) / exp_sum;
                tensor_put_elem(&output_tensor, (size_t[]){i, j}, new_elem);
            }
        }
    }
    return output_tensor;
}


// Tensor tensor_scale(Tensor *input, Tensor *scale){
//     assert(scale->ndim == 2);
//     Tensor output_tensor = tensor_init(
//         NULL, 
//         input->shape, 
//         input->ndim, 
//         input->dtype, 
//         input->requires_grad, 
//         false
//     );

//     size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
//     size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
//     size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


//     for(size_t i = 0; i < batch_size; i++){
//         for(size_t j = 0; j < rows; j++){
//             double scale_factor =  tensor_get_elem(scale, (size_t[]){j, i});
//             for(size_t k = 0; k < cols; k++){
//                 double old_elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k});
//                 double new_elem = old_elem * scale_factor;
//                 tensor_put_elem(&output_tensor, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k}, new_elem);
//             }
//         }
//     }
//     return output_tensor;
// }

Tensor tensor_elementwise_scale(Tensor *input, double elem){
    Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){input->shape[0],  
        input->shape[1]}, 
        2,
        input->dtype,
        input->requires_grad,
        true
    );


    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];
    for(size_t i = 0; i < input->shape[0]; i++){
        for(size_t j = 0; j < input->shape[1]; j++){
            double old_elem = tensor_get_elem(input, (size_t[]){i, j});
            double new_elem = old_elem * elem;
            tensor_put_elem(&output_tensor, (size_t[]){i, j}, new_elem);
        }
    }
    return output_tensor;
}

Tensor tensor_vector_scale(Tensor *input, Tensor *vector){
    assert(vector->ndim == 2);
    Tensor output_tensor = tensor_init(
        NULL, 
        input->shape, 
        input->ndim, 
        input->dtype, 
        input->requires_grad, 
        false
    );

    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double scale_factor =  tensor_get_elem(vector, (size_t[]){j, i});
            for(size_t k = 0; k < cols; k++){
                double old_elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k});
                double new_elem = old_elem * scale_factor;
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k}, new_elem);
            }
        }
    }
    return output_tensor;
}

Tensor tensor_concat(Tensor *tensors, size_t no_of_tensors, size_t dim){
    //this function doesn't work on batch of tensors
    if(dim == 1){
        //concat across columns
        size_t new_cols = tensors[0].shape[1];
        for(size_t i = 1; i < no_of_tensors; i++){
            assert(tensors[i].shape[0] == tensors[i-1].shape[0]);
            new_cols += tensors[i].shape[1];
        }
        Tensor output_tensor = tensor_init(
            NULL, 
            (size_t[]){tensors[0].shape[0], new_cols}, 
            2,
            tensors[0].dtype,
            tensors[0].requires_grad,
            false
        );
        for(size_t t = 0; t < no_of_tensors; t++){
            for(size_t i = 0; i < tensors[t].shape[0]; i++){
                for(size_t j = 0; j < tensors[t].shape[1]; j++){
                    double elem = tensor_get_elem(&tensors[t], (size_t[]){i, j});
                    tensor_put_elem(&output_tensor, (size_t[]){i, j + (t * no_of_tensors)}, elem);
                }
            }
        }
        return output_tensor;
    }
    return (Tensor){};
}

Tensor *tensor_chunk(Tensor *tensor, size_t chunks, size_t dim){
    assert(tensor->ndim == 2);
    Tensor *output_tensors = calloc(chunks, sizeof(Tensor));
    if(dim == 1){
        assert(tensor->shape[1] % chunks == 0);
        //split columns
        size_t cols = tensor->shape[1] / chunks;

        for(size_t chunk = 0; chunk < chunks; chunk++){
            Tensor output_tensor = tensor_init(
                NULL, 
                (size_t[]){tensor->shape[0], cols}, 
                2,
                tensor->dtype,
                tensor->requires_grad,
                false
            );

            for(size_t i = 0; i < tensor->shape[0]; i++){
                for(size_t j = 0; j < cols; j++){
                    double elem = tensor_get_elem(tensor, (size_t[]){i, j + (chunk * chunks)});
                    tensor_put_elem(&output_tensor, (size_t[]){i, j}, elem);
                }
            }
            output_tensors[chunk] = output_tensor;
        }
    }
    return output_tensors;
}


Tensor tensor_cat(Tensor **tensors, size_t len){
    
    Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){tensors[0]->shape[0],  
        tensors[0]->shape[1]*len}, 
        2,
        tensors[0]->dtype,
        tensors[0]->requires_grad,
        false
    );
    size_t out_col = 0;
    for(size_t i = 0; i < len; i++){
        for(size_t j = 0; j < tensors[i]->shape[0]; j++){
            for(size_t k = 0; k < tensors[i]->shape[1]; k++){
                double elem = tensor_get_elem(tensors[i], (size_t[]){j, k});
                tensor_put_elem(&output_tensor, (size_t[]){j, k+out_col}, elem);
            }
        }
        out_col += tensors[i]->shape[1];
    }
    return output_tensor;
}

Tensor tensor_arange(int start, int end, int steps){
    //returns 1d tensor only
    Tensor output_tensor = tensor_init(
        NULL, 
        (size_t[]){1, (int)((end - start)/steps)}, 
        2,
        DTYPE_INT,
        false,
        false
    );
    for(size_t i = start; i <=end; i += steps){
        tensor_put_elem(&output_tensor, (size_t[]){0, (double)i}, i);
    }
    return output_tensor;
}
void tensor_mat_mul(const Tensor *tensor1, const Tensor *tensor2, Tensor *output_tensor, size_t batch_dim){
    size_t t1_rows = 0;
    size_t t1_cols = 0;
    size_t t2_rows = 0;
    size_t t2_cols = 0;
    size_t out_rows = 0;
    size_t out_cols = 0;
    if(output_tensor->ndim == 3){
        t1_rows = tensor1->shape[1];
        t1_cols = tensor1->shape[2];
        t2_rows = tensor2->shape[0];
        t2_cols = tensor2->shape[1];
        out_rows = output_tensor->shape[1];
        out_cols = output_tensor->shape[2];
    }
    else if(output_tensor->ndim == 2){
        t1_rows = tensor1->shape[0];
        t1_cols = tensor1->shape[1];
        t2_rows = tensor2->shape[0];
        t2_cols = tensor2->shape[1];
        out_rows = output_tensor->shape[0];
        out_cols = output_tensor->shape[1];
    }

    for(size_t i = 0; i < out_rows; i++){
        for(size_t j = 0; j < out_cols; j++){
            double result = 0;
            for(size_t k = 0; k < t1_cols; k++){
                double elem1 = 0;
                if(output_tensor->ndim == 3) elem1 = tensor_get_elem(tensor1, (size_t[]){batch_dim, i, k});
                else if(output_tensor->ndim == 2) elem1 = tensor_get_elem(tensor1, (size_t[]){i, k});
                double elem2 = tensor_get_elem(tensor2, (size_t[]){k, j});
                result += elem1 * elem2;
            }
            if(output_tensor->ndim == 3) tensor_put_elem(output_tensor, (size_t[]){batch_dim, i, j}, result);
            else if(output_tensor->ndim == 2)  tensor_put_elem(output_tensor, (size_t[]){i, j}, result);
        }
    }
}


Tensor tensor_dot_product(const Tensor *tensor1, const Tensor *tensor2){
    //assert(tensor1->ndim == tensor2->ndim);
    // tensor_print(tensor1, "================== TENSOR 1 ==================");
    // tensor_print(tensor2, "================== TENSOR 2 ==================");
    if(tensor1->ndim == 3 && tensor2->ndim == 2){
        assert(tensor1->shape[2] == tensor2->shape[0]);
        size_t batch_size = tensor1->shape[0];
        Tensor output_tensor =  tensor_init(
            NULL, 
            (size_t[]){batch_size, tensor1->shape[1], tensor2->shape[1]}, tensor1->ndim,
            tensor1->dtype,
            tensor1->requires_grad,
            false
        );
        for(size_t b = 0; b < batch_size; b++){
            tensor_mat_mul(tensor1, tensor2, &output_tensor, b);
        }
        return output_tensor;
    }
    else if(tensor1->ndim == 2 && tensor2->ndim == 2){
        Tensor output_tensor =  tensor_init(
            NULL, 
            (size_t[]){tensor1->shape[0], tensor2->shape[1]}, tensor1->ndim,
            tensor1->dtype,
            tensor1->requires_grad,
            false
        );
        tensor_mat_mul(tensor1, tensor2, &output_tensor, 0);
        return output_tensor;
    }
    return (Tensor){};
}




// Tensor tensor_dot_product(const Tensor *tensor1, const Tensor *tensor2){
//     assert(tensor1->shape[1] == tensor2->shape[0]);
//     Tensor output_tensor =  tensor_init(
//         NULL, 
//         (size_t[]){tensor1->shape[0],  
//         tensor2->shape[1]}, 
//         2,
//         tensor1->dtype,
//         tensor1->requires_grad,
//         true
//     );
//     size_t t1_rows = tensor1->shape[0];
//     size_t t1_cols = tensor1->shape[1];
//     size_t t2_rows = tensor2->shape[0];
//     size_t t2_cols = tensor2->shape[1];
//     size_t out_rows = output_tensor.shape[0];
//     size_t out_cols = output_tensor.shape[1];
//     for(size_t i = 0; i < out_rows; i++){
//         for(size_t j = 0; j < out_cols; j++){
//             double result = 0;
//             for(size_t k = 0; k < t1_cols; k++){
//                 double elem1 = tensor_get_elem(tensor1, (size_t[]){i, k});
//                 double elem2 = tensor_get_elem(tensor2, (size_t[]){k, j});
//                 result += elem1 * elem2;
//             }
//             tensor_put_elem(&output_tensor, i, j, result);
//         }
//     }
//     return output_tensor;
// }

Tensor tensor_add(Tensor *tensor1, Tensor *tensor2){
    Tensor outout_tensor = tensor_init(
        tensor1->data, 
        tensor1->shape, 
        tensor1->ndim, 
        tensor1->dtype, 
        tensor1->requires_grad, 
        false
    );
    for(size_t i = 0; i < outout_tensor.size; i++){
        ((double*)outout_tensor.data)[i] = ((double*)outout_tensor.data)[i] + ((double*)tensor2->data)[i];
    }
    return outout_tensor;
}

Tensor tensor_elementwise_add(Tensor *tensor, double val){

    Tensor outout_tensor = tensor_init(
        tensor->data, 
        tensor->shape, 
        tensor->ndim, 
        tensor->dtype, 
        tensor->requires_grad, 
        false
    );

    size_t batch_size = tensor->ndim == 3 ? tensor->shape[0]: 1;
    size_t rows = tensor->ndim == 3 ? tensor->shape[1]: tensor->shape[0];
    size_t cols = tensor->ndim == 3 ? tensor->shape[2]: tensor->shape[1];

    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(tensor, tensor->ndim ==3 ? (size_t[]){i, j, k}:  (size_t[]){j, k});
                tensor_put_elem(tensor, tensor->ndim ==3 ? (size_t[]){i, j, k}:  (size_t[]){j, k}, val * elem);
            }
        }
    }
    return outout_tensor;
}


Tensor tensor_vector_add(Tensor *input, Tensor *vector){
    assert(vector->ndim == 2);
    Tensor output_tensor = tensor_init(
        NULL, 
        input->shape, 
        input->ndim, 
        input->dtype, 
        input->requires_grad, 
        false
    );

    size_t batch_size = input->ndim == 3 ? input->shape[0]: 1;
    size_t rows = input->ndim == 3 ? input->shape[1]: input->shape[0];
    size_t cols = input->ndim == 3 ? input->shape[2]: input->shape[1];


    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double scale_factor =  tensor_get_elem(vector, (size_t[]){j, i});
            for(size_t k = 0; k < cols; k++){
                double old_elem = tensor_get_elem(input, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k});
                double new_elem = old_elem + scale_factor;
                tensor_put_elem(&output_tensor, input->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k}, new_elem);
            }
        }
    }
    return output_tensor;
}

Tensor tensor_tril(Tensor *input, double elem){
    //assert(ndim == 2);
    Tensor output_tensor = tensor_init(
        NULL, 
        input->shape, 
        input->ndim, 
        input->dtype, 
        input->requires_grad, 
        false
    );


    // Tensor new_tensor = tensor_init(NULL, shape, ndim, dtype, false, false);
    for(size_t i = 0; i < input->shape[0]; i++){
        for(size_t j = 0; j < input->shape[1]; j++){
            if(j > i) tensor_put_elem(input, (size_t[]){i,j}, elem);
            // else tensor_put_elem(&new_tensor, (size_t[]){i,j}, 0);
        }
    }
    return output_tensor;
}

void tensor_masked_fill(Tensor *tensor, double mask, double fill){
    assert(tensor->ndim == 2);
    for(size_t i = 0; i < tensor->shape[0]; i++){
        for(size_t j = 0; j < tensor->shape[1]; j++){
            double elem = tensor_get_elem(tensor, (size_t[]){i, j});
            if(elem == mask){
                tensor_put_elem(tensor, (size_t[]){i, j}, fill);
            }
        }
    }
}



void tensor_copy_row_data(Tensor *dest_tensor, size_t batch_id, size_t row_id, Tensor *src_tensor, size_t src_row, size_t no_of_items){
    size_t dest_index = batch_id * dest_tensor->stride[0] + row_id * dest_tensor->stride[1];
    size_t src_index =  src_row * src_tensor->stride[0];
    // printf("dest index %zu, ", dest_index);
    // printf("src row: %zu\n", src_index);
    void *dest = &((double*)dest_tensor->data)[dest_index];
    void *src =  &((double*)src_tensor->data)[src_index];
    memcpy(dest, src, no_of_items * src_tensor->elem_size);
}

Tensor tensor_mean_var(Tensor *x){

    Tensor output_tensor = tensor_init(
        NULL, 
        x->ndim ==3 ? (size_t[]){x->shape[0], x->shape[1], 2}:  (size_t[]){x->shape[0], 2}, 
        x->ndim, 
        x->dtype, 
        x->requires_grad, 
        false
    );
    size_t batch_size = x->ndim == 3 ? x->shape[0]: 1;
    size_t rows = x->ndim == 3 ? x->shape[1]: x->shape[0];
    size_t cols = x->ndim == 3 ? x->shape[2]: x->shape[1];
    //printf("batch_size: %zu, rows: %zu, cols: %zu, x->ndim: %zu\n", batch_size, rows, cols, x->ndim);
    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double mean = 0;
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(x, x->ndim ==3 ? (size_t[]){i, j, k}:  (size_t[]){j, k});
                mean += elem;
            }
            mean = mean / cols;
            printf("%f\n", mean);

            double variance = 0;
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(x, x->ndim ==3 ? (size_t[]){i, j, k}:  (size_t[]){j, k});
                double squared_deviation = pow(elem - mean, 2);
                //double sqrt_squared_deviation = sqrt(squared_deviation);
                variance += squared_deviation;
                //printf("elem: %.3f, mean: %.3f, elem-mean: %.3f, squared_deviation: %.3f, sqrt_squared_deviation: %.3f\n", elem, mean, elem - mean, squared_deviation, sqrt_squared_deviation);
            }
            variance = variance / cols; 
            tensor_put_elem(&output_tensor, x->ndim ==3 ? (size_t[]){i, j, 0}:  (size_t[]){j, 0}, mean);
            tensor_put_elem(&output_tensor, x->ndim ==3 ? (size_t[]){i, j, 1}:  (size_t[]){j, 1}, variance);
        }
    }
    return output_tensor;
}

Tensor tensor_norm(Tensor *x, Tensor *mean_var_tensor, double eps){
    assert(x->ndim == mean_var_tensor->ndim);
    assert(mean_var_tensor->shape[mean_var_tensor->ndim-1] == 2);
    assert(x->shape[0] == mean_var_tensor->shape[0]);
    if(x->ndim == 3) assert(x->shape[1] == mean_var_tensor->shape[1]);

    
    Tensor output_tensor = tensor_init(
        NULL, 
        x->shape,
        x->ndim, 
        x->dtype, 
        x->requires_grad, 
        false
    );

    size_t batch_size = x->ndim == 3 ? x->shape[0]: 1;
    size_t rows = x->ndim == 3 ? x->shape[1]: x->shape[0];
    size_t cols = x->ndim == 3 ? x->shape[2]: x->shape[1];
    for(size_t i = 0; i < batch_size; i++){
        for(size_t j = 0; j < rows; j++){
            double mean = tensor_get_elem(mean_var_tensor, x->ndim == 3 ? (size_t[]){i, j, 0}: (size_t[]){j, 0});
            double var = tensor_get_elem(mean_var_tensor, x->ndim == 3 ? (size_t[]){i, j, 1}: (size_t[]){j, 1});
            //printf("mean: %.2f, var: %.2f\n", mean, var);
            for(size_t k = 0; k < cols; k++){
                double elem = tensor_get_elem(x, x->ndim == 3? (size_t[]){i, j, k}: (size_t[]){j, k});
                double norm_elem = (elem - mean) / sqrt(var + eps);
                tensor_put_elem(&output_tensor, x->ndim == 3 ? (size_t[]){i, j, k}: (size_t[]){j, k}, norm_elem);
            }
        }
    }
    return output_tensor;
}

void tensor_print(const Tensor *tensor, const char *heading){
    printf("\n============== %s ==================\n", heading);
    printf("size:           %zu\n", tensor->size);
    printf("ndim:           %zu\n", tensor->ndim);
    
    printf("shape:          ( ");
    for(size_t i = 0; i < tensor->ndim; i++){
        printf("%zu, ", tensor->shape[i]);
    }
    printf(" )\n");
    printf("stride:         ( ");
    for(size_t i = 0; i < tensor->ndim; i++){
        printf("%zu, ", tensor->stride[i]);
    }
    printf(" )\n");
    printf("elem_size:      %zu\n", tensor->elem_size);
    printf("requires_grad:  %s\n", tensor->requires_grad ? "true" : "false");
    printf("data:\n");
    
    if(tensor->ndim == 3){
        printf("[\n");
        for(size_t b = 0; b < tensor->shape[0]; b++){
            for(size_t i = 0; i < tensor->shape[1]; i++){
                printf("    [ ");
                for(size_t j = 0; j < tensor->shape[2]; j++){
                    double elem = tensor_get_elem(tensor, (size_t[]){b, i, j});
                    if(tensor->dtype == DTYPE_DOUBLE) printf("%10.2f ", elem);
                    else if(tensor->dtype == DTYPE_INT) printf("%d ", (int)elem);
                }
                printf(" ]\n");
            }
            if(b <= tensor->shape[0] - 2) printf("\n\n");
        }
        printf("]\n");
    }
    else if(tensor->ndim == 2){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            printf("[ ");
            for(size_t j = 0; j < tensor->shape[1]; j++){
                double elem = tensor_get_elem(tensor, (size_t[]){i, j});
                if(tensor->dtype == DTYPE_DOUBLE) printf("%10.2f ", elem);
                else if(tensor->dtype == DTYPE_INT) printf("%d    ", (int)elem);
            }
            printf(" ]\n");
        }
    }
    printf("\n");
}


void tensor_write_fp(const Tensor *tensor, FILE *fptr){
    fprintf(fptr, "size,%zu\n", tensor->size);
    fprintf(fptr, "ndim,%zu\n", tensor->ndim);
    fprintf(fptr, "shape,");
    for(size_t i = 0; i < tensor->ndim; i++){
        fprintf(fptr, "%zu,", tensor->shape[i]);
    }
    fprintf(fptr, "\nstride,");
    for(size_t i = 0; i < tensor->ndim; i++){
        fprintf(fptr, "%zu,", tensor->stride[i]);
    }
    fprintf(fptr, "\nelem_size,%zu\n", tensor->elem_size);
    fprintf(fptr, "requires_grad,%d\n\n", tensor->requires_grad);

    if(tensor->ndim == 3){
        for(size_t b = 0; b < tensor->shape[0]; b++){
            for(size_t i = 0; i < tensor->shape[1]; i++){
                for(size_t j = 0; j < tensor->shape[2]; j++){
                    double elem = tensor_get_elem(tensor, (size_t[]){b, i, j});
                    fprintf(fptr, ",%5.2f", elem);
                }
                fprintf(fptr, "\n");
            }
            if(b <= tensor->shape[0] - 2)  fprintf(fptr, "\n");
        }
        fprintf(fptr, "\n");
    }
    else if(tensor->ndim == 2){
        for(size_t i = 0; i < tensor->shape[0]; i++){
            for(size_t j = 0; j < tensor->shape[1]; j++){
                double elem = tensor_get_elem(tensor, (size_t[]){i, j});
                fprintf(fptr, ",%5.2f", elem);
            }
            fprintf(fptr, "\n");
        }
    }
    //fprintf(fptr, "\n");
}

void tensor_write(const Tensor *tensor, char *filename){
    FILE *fptr = fopen(filename, "w");
    if(fptr == NULL){
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    tensor_write_fp(tensor, fptr);
}


size_t tensor_dtype_size(const DataType dtype){
    switch (dtype){
        case DTYPE_DOUBLE: return sizeof(double);
        case DTYPE_INT: return sizeof(int);
    }
}

