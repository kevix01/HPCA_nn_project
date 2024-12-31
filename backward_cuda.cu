//
// Created by kevin on 24/12/24.
//

#include <cuda_runtime.h> // CUDA runtime API
#include "backward_cuda.h" // Header for CUDA backward pass

#include <iostream> // For input/output
#include <vector> // For std::vector

// ReLU derivative function (device code)
__device__ float reluDerivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f; // Returns 1 if x > 0, else 0
}

// Sigmoid derivative function (device code)
__device__ float sigmoidDerivative(float x) {
    float sigmoid = 1.0f / (1.0f + expf(-x)); // Compute sigmoid
    return sigmoid * (1.0f - sigmoid); // Return sigmoid derivative
}

// Activation derivative selector (device code)
__device__ float activateDerivative(float x, ActivationFunctionType act_type) {
    if (act_type == RELU) {
        return reluDerivative(x); // Use ReLU derivative
    } else if (act_type == SIGMOID) {
        return sigmoidDerivative(x); // Use Sigmoid derivative
    }
    return 1.0f; // Default derivative (identity function)
}

// CUDA kernel to compute deltas and update biases
__global__ void computeDeltasAndBiases(const float* grad, const float* outputCache, float* deltas, float* biases,
                                       int batchSize, int outputSize, float learningRate, ActivationFunctionType act_type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Thread index for output neurons
    if (i >= outputSize) return; // Exit if out of bounds

    float avg_delta = 0.0f; // Accumulator for average delta
    for (int k = 0; k < batchSize; ++k) {
        int index = k * outputSize + i; // Index for gradient and output cache
        deltas[index] = grad[index] * activateDerivative(outputCache[index], act_type); // Compute delta
        avg_delta += deltas[index]; // Accumulate delta
    }
    avg_delta /= batchSize; // Compute average delta

    biases[i] -= learningRate * avg_delta; // Update bias
}

// CUDA kernel to update weights and compute gradient of the input
__global__ void updateWeightsAndGradInput(const float* deltas, const float* inputCache, float* weights, float* gradInput,
                                          int inputSize, int outputSize, int batchSize, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Thread index for weight matrix
    int i = idx / inputSize; // Row index (output neuron)
    int j = idx % inputSize; // Column index (input neuron)

    if (i >= outputSize || j >= inputSize) return; // Exit if out of bounds

    float weight_step = 0.0f; // Accumulator for weight update

    for (int k = 0; k < batchSize; ++k) {
        int gradIndex = k * outputSize + i; // Index for delta
        int inputIndex = k * inputSize + j; // Index for input cache

        weight_step += deltas[gradIndex] * inputCache[inputIndex]; // Compute weight step
        atomicAdd(&gradInput[inputIndex], deltas[gradIndex] * weights[i * inputSize + j]); // Accumulate gradient of input
    }

    weight_step /= batchSize; // Compute average weight step

    weights[i * inputSize + j] -= learningRate * weight_step; // Update weight
}

// Host function to perform backward pass on the GPU
std::vector<float> backward_cuda(const std::vector<float>& grad, const std::vector<float>& outputCache,
                                 const std::vector<float>& inputCache, std::vector<float>& weights,
                                 std::vector<float>& biases, int outputSize, int inputSize,
                                 int batchSize, float learningRate, ActivationFunctionType act_type, int block_size) {
    int gradFlatSize = batchSize * outputSize; // Size of flattened gradient
    int inputFlatSize = batchSize * inputSize; // Size of flattened input cache

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
    int numBlocksDeltas = (outputSize + block_size - 1) / block_size; // Blocks for deltas kernel
    int numBlocksWeights = ((outputSize * inputSize) + block_size - 1) / block_size; // Blocks for weights kernel

    // Launch kernel to compute deltas and update biases
    auto start_kernel_deltas = std::chrono::high_resolution_clock::now();
    computeDeltasAndBiases<<<numBlocksDeltas, block_size>>>(d_grad, d_outputCache, d_deltas, d_biases, batchSize, outputSize, learningRate, act_type);
    auto end_kernel_deltas = std::chrono::high_resolution_clock::now();
    elapsed_b_kernel_deltas += end_kernel_deltas - start_kernel_deltas; // Accumulate kernel execution time
    CHECK_CUDA_ERROR_B(cudaPeekAtLastError(), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaDeviceSynchronize(), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });

    // Launch kernel to update weights and compute gradient of the input
    auto start_kernel_weights = std::chrono::high_resolution_clock::now();
    updateWeightsAndGradInput<<<numBlocksWeights, block_size>>>(d_deltas, d_inputCache, d_weights, d_gradInput, inputSize, outputSize, batchSize, learningRate);
    auto end_kernel_weights = std::chrono::high_resolution_clock::now();
    elapsed_b_kernel_weights += end_kernel_weights - start_kernel_weights; // Accumulate kernel execution time
    CHECK_CUDA_ERROR_B(cudaPeekAtLastError(), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });
    CHECK_CUDA_ERROR_B(cudaDeviceSynchronize(), { cudaFree(d_grad); cudaFree(d_outputCache); cudaFree(d_inputCache); cudaFree(d_deltas); cudaFree(d_biases); cudaFree(d_weights); cudaFree(d_gradInput); });

    // Copy results back to host
    std::vector<float> gradInput(inputFlatSize); // Gradient of the input
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

    return gradInput; // Return gradient of the input
}
