#include "../include/tensor.h"
#include "../include/interface.h"
#include "../include/parameter.h"

parameter * create_parameter (tensor *tensor_ptr) {
    if (!tensor_ptr) {
        return NULL;
    }

    parameter* param = (parameter*)malloc(sizeof(parameter));
    if (!param) {
        perror("Failed to allocate parameter");
        return NULL;
    }

    param->tensor_ptr = tensor_ptr;
    param->first_moment = 0.0;
    param->second_moment = 0.0;
    return param;
}

void free_parameter(parameter *param) {
    if (param) {
        free(param);
    }
}