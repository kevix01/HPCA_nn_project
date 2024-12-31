//
// Created by kevin on 24/12/24.
//

#ifndef BACKWARD_CUDA_H
#define BACKWARD_CUDA_H

#include <chrono>
#include <vector>
#include "actf_type_cuda.h"

extern std::chrono::duration<double> elapsed_b_kernel_deltas;
extern std::chrono::duration<double> elapsed_b_kernel_weights;


#define CHECK_CUDA_ERROR_B(call, cleanup) do {                                   \
    cudaError_t err = call;                                                      \
    if (err != cudaSuccess) {                                                    \
        std::cerr << "CUDA Error in Backward: " << cudaGetErrorString(err)       \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
        cleanup;                                                                 \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while(0)



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

