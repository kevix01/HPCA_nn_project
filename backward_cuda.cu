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
    // printf("Average delta for neuron %d: %f\n", i, avg_delta);

    atomicAdd(&biases[i], -learningRate * avg_delta);
    //biases[i] -= avg_delta;
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

        // float delta = deltas[gradIndex];
        weight_step += deltas[gradIndex] * inputCache[inputIndex];
        //if (inputSize <= 4)
        //printf("Partial weight step for neuron with delta index %d, input value %f, sample %d and input %d: %f\n", gradIndex, inputCache[inputIndex], k, j, deltas[gradIndex] * inputCache[inputIndex]);
        atomicAdd(&gradInput[inputIndex], deltas[gradIndex] * weights[i * inputSize + j]);
    }
    //if (inputSize <= 4 && idx == 0) {
    // print all gradInput
    /*if (idx == 0) {
        for (int k = 0; k < batchSize; ++k) {
            for (int j = 0; j < inputSize; ++j) {
                printf("GradInput for sample %d and input %d: %f\n", k, j, gradInput[k * inputSize + j]);
            }
        }
    }*/
    //}
    weight_step /= batchSize;
    //if (inputSize >4)
    // printf("Weight step update for neuron %d, input %d: %f\n", i, j, weight_step);

    atomicAdd(&weights[i * inputSize + j], -learningRate * weight_step);
    //if (inputSize <=4)
    // printf("Weight value for neuron %d, input %d: %f\n", i, j, weights[i * inputSize + j]);
    //weights[i * inputSize + j] -= learningRate * weight_step;
}

/*__global__ void updateWeightsAndGradInput(const float* deltas, const float* inputCache, double* weights, double* gradInput,
                                          int inputSize, int outputSize, int batchSize, float learningRate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / inputSize;
    int j = idx % inputSize;

    if (i >= outputSize || j >= inputSize) return;

    double weight_step = 0.0;

    // Perform double precision accumulation for weight updates
    for (int k = 0; k < batchSize; ++k) {
        int gradIndex = k * outputSize + i;
        int inputIndex = k * inputSize + j;

        weight_step += static_cast<double>(deltas[gradIndex]) * static_cast<double>(inputCache[inputIndex]);
        atomicAdd(&gradInput[inputIndex], static_cast<double>(deltas[gradIndex]) * weights[i * inputSize + j]);
    }

    weight_step /= static_cast<double>(batchSize);
    atomicAdd(&weights[i * inputSize + j], -static_cast<double>(learningRate) * weight_step);
}*/

/*std::vector<float> backward_cuda(const std::vector<float>& grad, const std::vector<float>& outputCache,
                                 const std::vector<float>& inputCache, std::vector<float>& weights, std::vector<float>& biases,
                                 int outputSize, int inputSize, int batchSize, float learningRate, ActivationFunctionType act_type) {
    int gradFlatSize = batchSize * outputSize;
    int inputFlatSize = batchSize * inputSize;

    float *d_grad, *d_outputCache, *d_inputCache, *d_deltas, *d_biases;
    double *d_weights, *d_gradInput;

    // Allocate device memory
    cudaMalloc(&d_grad, gradFlatSize * sizeof(float));
    cudaMalloc(&d_outputCache, gradFlatSize * sizeof(float));
    cudaMalloc(&d_inputCache, inputFlatSize * sizeof(float));
    cudaMalloc(&d_deltas, gradFlatSize * sizeof(float));
    cudaMalloc(&d_biases, outputSize * sizeof(float));
    cudaMalloc(&d_weights, outputSize * inputSize * sizeof(double)); // Double precision for weights
    cudaMalloc(&d_gradInput, inputFlatSize * sizeof(double)); // Double precision for gradients

    // Check if memory allocation was successful
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error during memory allocation: " << cudaGetErrorString(err) << std::endl;
        return {};
    }

    // Copy data to device (host -> device)
    cudaMemcpy(d_grad, grad.data(), gradFlatSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputCache, outputCache.data(), gradFlatSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputCache, inputCache.data(), inputFlatSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), outputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice); // Copy weights as float
    cudaMemcpy(d_biases, biases.data(), outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_gradInput, 0, inputFlatSize * sizeof(double)); // Zero out gradInput in double precision

    // Kernel launch configurations
    int blockSize = 16;
    int numBlocksDeltas = (outputSize + blockSize - 1) / blockSize;
    int numBlocksWeights = ((outputSize * inputSize) + blockSize - 1) / blockSize;

    // Launch kernel
    computeDeltasAndBiases<<<numBlocksDeltas, blockSize>>>(d_grad, d_outputCache, d_deltas, d_biases, batchSize, outputSize, learningRate, act_type);
    updateWeightsAndGradInput<<<numBlocksWeights, blockSize>>>(d_deltas, d_inputCache, d_weights, d_gradInput, inputSize, outputSize, batchSize, learningRate);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after kernel execution: " << cudaGetErrorString(err) << std::endl;
        return {};
    }

    // Copy results back to host (device -> host)
    std::vector<float> gradInput(inputFlatSize);
    cudaMemcpy(gradInput.data(), d_gradInput, inputFlatSize * sizeof(double), cudaMemcpyDeviceToHost); // Copy gradInput in double, convert on host
    cudaMemcpy(weights.data(), d_weights, outputSize * inputSize * sizeof(double), cudaMemcpyDeviceToHost); // Copy weights in double, convert on host
    cudaMemcpy(biases.data(), d_biases, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert double precision to float for final results
    for (int i = 0; i < gradInput.size(); ++i) {
        gradInput[i] = static_cast<float>(gradInput[i]);
    }
    for (int i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>(weights[i]);
    }

    // Free device memory (ensure you free everything that was allocated)
    cudaFree(d_grad);
    cudaFree(d_outputCache);
    cudaFree(d_inputCache);
    cudaFree(d_deltas);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_gradInput);

    return gradInput;
}*/



std::vector<float> backward_cuda(const std::vector<float>& grad, const std::vector<float>& outputCache,
                                 const std::vector<float>& inputCache, std::vector<float>& weights,
                                 std::vector<float>& biases, int outputSize, int inputSize,
                                 int batchSize, float learningRate, ActivationFunctionType act_type) {
    int gradFlatSize = batchSize * outputSize;
    int inputFlatSize = batchSize * inputSize;

    /*if (outputSize == 4) {
        // print all grad given
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                std::cout << "Grad for sample " << i << " and output " << j << ": " << grad[i * outputSize + j] << std::endl;
            }
        }
    }*/

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
    int blockSize = 512;
    int numBlocksDeltas = (outputSize + blockSize - 1) / blockSize;
    int numBlocksWeights = ((outputSize * inputSize) + blockSize - 1) / blockSize;

    // Launch kernels
    computeDeltasAndBiases<<<numBlocksDeltas, blockSize>>>(d_grad, d_outputCache, d_deltas, d_biases, batchSize, outputSize, learningRate, act_type);
    // Print the deltas
    /*std::vector<float> deltas(gradFlatSize);
    cudaMemcpy(deltas.data(), d_deltas, gradFlatSize * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < gradFlatSize; i++) {
        std::cout << "Delta " << i << ": " << deltas[i] << std::endl;
    }*/
    // print the biases
    /*if (inputSize > 4) {
        std::vector<float> biases_host(outputSize);
        cudaMemcpy(biases_host.data(), d_biases, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < outputSize; i++) {
            std::cout << "Bias " << i << ": " << biases_host[i] << std::endl;
        }
    }*/
    /*dim3 blockDim(16, 16); // 32x32 threads per block
    dim3 gridDim((outputSize + blockDim.x - 1) / blockDim.x,
                 (inputSize + blockDim.y - 1) / blockDim.y);
    updateWeightsAndGradInput<<<gridDim, blockDim>>>(d_deltas, d_inputCache, d_weights, d_gradInput,
                                                     inputSize, outputSize, batchSize, learningRate);*/

    updateWeightsAndGradInput<<<numBlocksWeights, blockSize>>>(d_deltas, d_inputCache, d_weights, d_gradInput, inputSize, outputSize, batchSize, learningRate);

    // Synchronize
    cudaDeviceSynchronize();

    // Copy results back to host
    std::vector<float> gradInput(inputFlatSize);
    cudaMemcpy(gradInput.data(), d_gradInput, inputFlatSize * sizeof(float), cudaMemcpyDeviceToHost);
    // Print the gradInput
    /*if (inputSize <= 4) {
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                std::cout << "GradInput for sample " << i << " and input " << j << ": " << gradInput[i * inputSize + j] << std::endl;
            }
        }
    }*/
    cudaMemcpy(weights.data(), d_weights, outputSize * inputSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases.data(), d_biases, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    // print biases
    /*if (outputSize <= 4) {
        for (int i = 0; i < outputSize; i++) {
            std::cout << "Bias " << i << ": " << biases[i] << std::endl;
        }
    }*/
    // print weights
    /*if (outputSize <= 1) {
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                std::cout << "Weight " << i << " " << j << ": " << weights[i * inputSize + j] << std::endl;
            }
        }
    }*/

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






