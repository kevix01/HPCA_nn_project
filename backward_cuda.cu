//
// Created by kevin on 24/12/24.
//

#include <cuda_runtime.h>
#include "backward_cuda.h"

#include <iostream>
#include <vector>

__device__ float reluDerivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

__device__ float sigmoidDerivative(float x) {
    float sigmoid = 1.0f / (1.0f + expf(-x));
    return sigmoid * (1.0f - sigmoid);
}

__device__ float activateDerivative(float x, ActivationFunctionType act_type) {
    if (act_type == RELU) {
        return reluDerivative(x);
    } else if (act_type == SIGMOID) {
        return sigmoidDerivative(x);
    }
    return 1.0f;
}

__global__ void computeDeltasAndBiases(const float* grad, const float* outputCache, float* deltas, float* biases,
                                       int batchSize, int outputSize, float learningRate, ActivationFunctionType act_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outputSize) return;

    float avg_delta = 0.0f;
    for (int k = 0; k < batchSize; ++k) {
        int index = k * outputSize + i;
        deltas[index] = grad[index] * activateDerivative(outputCache[index], act_type);
        avg_delta += deltas[index];
    }
    avg_delta /= batchSize;

    biases[i] -= learningRate * avg_delta;
}

__global__ void updateWeightsAndGradInput(const float* deltas, const float* inputCache, float* weights, float* gradInput,
                                          int inputSize, int outputSize, int batchSize, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / inputSize;
    int j = idx % inputSize;

    if (i >= outputSize || j >= inputSize) return;

    float weight_step = 0.0f;

    for (int k = 0; k < batchSize; ++k) {
        int gradIndex = k * outputSize + i;
        int inputIndex = k * inputSize + j;

        weight_step += deltas[gradIndex] * inputCache[inputIndex];
        atomicAdd(&gradInput[inputIndex], deltas[gradIndex] * weights[i * inputSize + j]);
    }

    weight_step /= batchSize;

    weights[i * inputSize + j] -= learningRate * weight_step;
}

std::vector<float> backward_cuda(const std::vector<float>& grad, const std::vector<float>& outputCache,
                                 const std::vector<float>& inputCache, std::vector<float>& weights,
                                 std::vector<float>& biases, int outputSize, int inputSize,
                                 int batchSize, float learningRate, ActivationFunctionType act_type, int block_size) {
    int gradFlatSize = batchSize * outputSize;
    int inputFlatSize = batchSize * inputSize;

    float *d_grad = nullptr, *d_outputCache = nullptr, *d_inputCache = nullptr;
    float *d_deltas = nullptr, *d_biases = nullptr, *d_weights = nullptr, *d_gradInput = nullptr;

    // Allocate device memory
    CHECK_CUDA_ERROR_B(cudaMalloc(&d_grad, gradFlatSize * sizeof(float)), {});
    CHECK_CUDA_ERROR_B(cudaMalloc(&d_outputCache, gradFlatSize * sizeof(float)), { cudaFree(d_grad); });
    CHECK_CUDA_ERROR_B(cudaMalloc(&d_inputCache, inputFlatSize * sizeof(float)), { cudaFree(d_grad); cudaFree(d_outputCache); });
    CHECK_CUDA_ERROR_B(cudaMalloc(&d_deltas, gradFlatSize * sizeof(float)), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); });
    CHECK_CUDA_ERROR_B(cudaMalloc(&d_biases, outputSize * sizeof(float)), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); });
    CHECK_CUDA_ERROR_B(cudaMalloc(&d_weights, outputSize * inputSize * sizeof(float)), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); });
    CHECK_CUDA_ERROR_B(cudaMalloc(&d_gradInput, inputFlatSize * sizeof(float)), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); });

    // Copy data to device
    CHECK_CUDA_ERROR_B(cudaMemcpy(d_grad, grad.data(), gradFlatSize * sizeof(float), cudaMemcpyHostToDevice), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaMemcpy(d_outputCache, outputCache.data(), gradFlatSize * sizeof(float), cudaMemcpyHostToDevice), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaMemcpy(d_inputCache, inputCache.data(), inputFlatSize * sizeof(float), cudaMemcpyHostToDevice), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaMemcpy(d_weights, weights.data(), outputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaMemcpy(d_biases, biases.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaMemset(d_gradInput, 0, inputFlatSize * sizeof(float)), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });

    // Kernel launch configurations
    int numBlocksDeltas = (outputSize + block_size - 1) / block_size;
    int numBlocksWeights = ((outputSize * inputSize) + block_size - 1) / block_size;

    auto start_kernel_deltas = std::chrono::high_resolution_clock::now();
    // Launch kernels
    computeDeltasAndBiases<<<numBlocksDeltas, block_size>>>(d_grad, d_outputCache, d_deltas, d_biases, batchSize, outputSize, learningRate, act_type);
    auto end_kernel_deltas = std::chrono::high_resolution_clock::now();
    elapsed_b_kernel_deltas += end_kernel_deltas - start_kernel_deltas;
    CHECK_CUDA_ERROR_B(cudaPeekAtLastError(), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaDeviceSynchronize(), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });

    auto start_kernel_weights = std::chrono::high_resolution_clock::now();
    updateWeightsAndGradInput<<<numBlocksWeights, block_size>>>(d_deltas, d_inputCache, d_weights, d_gradInput, inputSize, outputSize, batchSize, learningRate);
    auto end_kernel_weights = std::chrono::high_resolution_clock::now();
    elapsed_b_kernel_weights += end_kernel_weights - start_kernel_weights;
    CHECK_CUDA_ERROR_B(cudaPeekAtLastError(), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaDeviceSynchronize(), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });

    // Copy results back to host
    std::vector<float> gradInput(inputFlatSize);
    CHECK_CUDA_ERROR_B(cudaMemcpy(gradInput.data(), d_gradInput, inputFlatSize * sizeof(float), cudaMemcpyDeviceToHost), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaMemcpy(weights.data(), d_weights, outputSize * inputSize * sizeof(float), cudaMemcpyDeviceToHost), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaMemcpy(biases.data(), d_biases, outputSize * sizeof(float), cudaMemcpyDeviceToHost), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });

    // Free device memory
    CHECK_CUDA_ERROR_B(cudaFree(d_grad), {});
    CHECK_CUDA_ERROR_B(cudaFree(d_outputCache), {});
    CHECK_CUDA_ERROR_B(cudaFree(d_inputCache), {});
    CHECK_CUDA_ERROR_B(cudaFree(d_deltas), {});
    CHECK_CUDA_ERROR_B(cudaFree(d_weights), {});
    CHECK_CUDA_ERROR_B(cudaFree(d_biases), {});
    CHECK_CUDA_ERROR_B(cudaFree(d_gradInput), {});

    return gradInput;
}
