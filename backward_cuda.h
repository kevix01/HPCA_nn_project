//
// Created by kevin on 24/12/24.
//

#ifndef BACKWARD_CUDA_H
#define BACKWARD_CUDA_H

#include <vector>
#include "actf_type_cuda.h"


// Function declaration for the CUDA backward pass
std::vector<float> backward_cuda(const std::vector<float>& grad,
                                 const std::vector<float>& outputCache,
                                 const std::vector<float>& inputCache,
                                 std::vector<float>& weights,
                                 std::vector<float>& biases,
                                 int outputSize,
                                 int inputSize,
                                 int batchSize,
                                 float learningRate,
                                 ActivationFunctionType act_type, int block_size);

#endif // BACKWARD_CUDA_H

