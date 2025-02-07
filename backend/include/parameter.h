#include "ops.h"

// representation of any trainable tensor (weights, biases, ..)
typedef struct parameter { 
    tensor *tensor_ptr;
    lemur_float first_moment;
    lemur_float second_moment;
} parameter;

parameter* create_parameter(tensor* tensor_ptr);
void free_parameter(parameter* param);