//
// Created by kevin on 24/12/24.
//

#include <cuda_runtime.h>
#include "backward_cuda.h"
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
                                       int batchSize, int outputSize, ActivationFunctionType act_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outputSize) return;

    float avg_delta = 0.0f;
    for (int k = 0; k < batchSize; ++k) {
        int index = k * outputSize + i;
        deltas[index] = grad[index] * activateDerivative(outputCache[index], act_type);
        avg_delta += deltas[index];
    }
    avg_delta /= batchSize;

    atomicAdd(&biases[i], -avg_delta);
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

        float delta = deltas[gradIndex];
        weight_step += delta * inputCache[inputIndex];
        atomicAdd(&gradInput[inputIndex], delta * weights[i * inputSize + j]);
    }

    weight_step /= batchSize;

    atomicAdd(&weights[i * inputSize + j], -learningRate * weight_step);
}

std::vector<float> backward_cuda(const std::vector<float>& grad, const std::vector<float>& outputCache,
                                 const std::vector<float>& inputCache, std::vector<float>& weights,
                                 std::vector<float>& biases, int outputSize, int inputSize,
                                 int batchSize, float learningRate, ActivationFunctionType act_type) {
    int gradFlatSize = batchSize * outputSize;
    int inputFlatSize = batchSize * inputSize;

    float *d_grad, *d_outputCache, *d_inputCache, *d_deltas, *d_biases, *d_weights, *d_gradInput;

    // Allocate device memory
    cudaMalloc(&d_grad, gradFlatSize * sizeof(float));
    cudaMalloc(&d_outputCache, gradFlatSize * sizeof(float));
    cudaMalloc(&d_inputCache, inputFlatSize * sizeof(float));
    cudaMalloc(&d_deltas, gradFlatSize * sizeof(float));
    cudaMalloc(&d_biases, outputSize * sizeof(float));
    cudaMalloc(&d_weights, outputSize * inputSize * sizeof(float));
    cudaMalloc(&d_gradInput, inputFlatSize * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_grad, grad.data(), gradFlatSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputCache, outputCache.data(), gradFlatSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputCache, inputCache.data(), inputFlatSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), outputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_gradInput, 0, inputFlatSize * sizeof(float));

    // Kernel launch configurations
    int blockSize = 256;
    int numBlocksDeltas = (outputSize + blockSize - 1) / blockSize;
    int numBlocksWeights = ((outputSize * inputSize) + blockSize - 1) / blockSize;

    // Launch kernels
    computeDeltasAndBiases<<<numBlocksDeltas, blockSize>>>(d_grad, d_outputCache, d_deltas, d_biases, batchSize, outputSize, act_type);
    updateWeightsAndGradInput<<<numBlocksWeights, blockSize>>>(d_deltas, d_inputCache, d_weights, d_gradInput, inputSize, outputSize, batchSize, learningRate);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy results back to host
    std::vector<float> gradInput(inputFlatSize);
    cudaMemcpy(gradInput.data(), d_gradInput, inputFlatSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weights.data(), d_weights, outputSize * inputSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases.data(), d_biases, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_grad);
    cudaFree(d_outputCache);
    cudaFree(d_inputCache);
    cudaFree(d_deltas);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_gradInput);

    return gradInput;
}

