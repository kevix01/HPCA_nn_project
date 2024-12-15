//
// Created by kevin on 06/12/24.
//

#include "linear_layer.h"
#include <cmath>
#include <iostream>
#include <random>
#include "cuda_matmul.h"

LinearLayer::LinearLayer(int inputSize, int outputSize, ActivationFunction activation, DeviceType device, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), activation(activation) {
    this->device = device;
    std::mt19937 gen(seed);  // Use the provided seed for reproducibility
    std::normal_distribution<> dis(0.0, 1.0 / std::sqrt(inputSize)); // Normal distribution initialization

    weights.resize(inputSize * outputSize);
    biases.resize(outputSize);

    for (auto& weight : weights) {
        weight = dis(gen);
    }

    for (auto& bias : biases) {
        bias = 0.0f; // Initialize biases to zero
    }
}


std::vector<float> LinearLayer::forward(const std::vector<float>& input) {
    inputCache = input;
    std::vector<float> output(outputSize);
    if (device == CPU){
        for (int i = 0; i < outputSize; ++i) {
            float sum = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += input[j] * weights[i * inputSize + j];
            }
            output[i] = activate(sum);
        }
    } else if (device == CUDA) {
        matMulCuda(input, output);
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] += biases[i];
            output[i] = activate(output[i]);
        }
    }
    outputCache = output;
    return output;
}

std::vector<float> LinearLayer::backward(const std::vector<float>& grad, float learningRate) {
    std::vector<float> gradInput(inputSize, 0.0f);

    for (int i = 0; i < outputSize; ++i) {
        float delta = grad[i] * activateDerivative(outputCache[i]);

        for (int j = 0; j < inputSize; ++j) {
            gradInput[j] += delta * weights[i * inputSize + j];
            weights[i * inputSize + j] -= learningRate * delta * inputCache[j];
        }

        biases[i] -= learningRate * delta;
    }

    return gradInput;
}

float LinearLayer::activate(float x) {
    if (activation == ActivationFunction::Sigmoid) {
        return 1.0f / (1.0f + std::exp(-x));
    } else if (activation == ActivationFunction::ReLU) {
        return std::max(0.0f, x);
    }
    return 0.0f;
}

float LinearLayer::activateDerivative(float x) {
    if (activation == ActivationFunction::Sigmoid) {
        float sig = activate(x);
        return sig * (1.0f - sig);
    } else if (activation == ActivationFunction::ReLU) {
        return x > 0.0f ? 1.0f : 0.0f;
    }
    return 0.0f;
}

void LinearLayer::matMulCuda(const std::vector<float>& inputs, std::vector<float>& outputs) {
    int M = outputs.size(); // Number of neurons
    int K = inputs.size(); // Number of inputs
    int N = outputs.size();
    // std::cout << "M: " << M << " K: " << K << " N: " << N << std::endl;
    // std::cout << outputs.size() << std::endl;
    // Replicate inputs for each neuron
    float *a = new float[K * M];
    for (int i = 0; i < M; ++i) {
        std::copy(inputs.begin(), inputs.end(), a + i * K);
    }

    // Flatten weights matrix appropriately
    float *b = new float[K * M]; // Now b should have size K * M
    float *ab = new float[M];

    // Flatten weights matrix in the correct order
    /*for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            b[j * M + i] = weights[i][j];
        }
    }*/
    // populate b with weights
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            b[j * M + i] = weights[i * K + j];
        }
    }

    // Debug prints for verification
    std::cout << "Replicated input a:" << std::endl;
    for (int i = 0; i < K * M; ++i) {
        std::cout << " " << a[i];
        if ((i + 1) % K == 0) std::cout << std::endl;
    }

    std::cout << "Flattened weights b:" << std::endl;
    for (int i = 0; i < K * M; ++i) {
        std::cout << " " << b[i];
        if ((i + 1) % M == 0) std::cout << std::endl;
    }

    matMul(a, b, ab, M, K, N);

    // Debug prints for verification
    /*std::cout << "Result ab:" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << " " << ab[i];
    }
    std::cout << std::endl;*/

    // Assign the result to outputs
    outputs.assign(ab, ab + M);

    // Clean up allocated memory
    delete[] a;
    delete[] b;
    delete[] ab;
}






