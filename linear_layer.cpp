//
// Created by kevin on 06/12/24.
//

#include "linear_layer.h"
#include "cuda_matmul.h"
#include <cmath>
#include <iostream>
#include <cstdlib>

LinearLayer::LinearLayer(int numInputs, int numNeurons, DeviceType device) : device(device) {
    weights.resize(numNeurons, std::vector<float>(numInputs));
    biases.resize(numNeurons);
    outputs.resize(numNeurons);
    deltas.resize(numNeurons);

    for (int i = 0; i < numNeurons; ++i) {
        for (int j = 0; j < numInputs; ++j) {
            weights[i][j] = static_cast<float>(rand()) / RAND_MAX; // Initialize weights randomly
        }
        biases[i] = static_cast<float>(rand()) / RAND_MAX; // Initialize biases randomly
    }
}

std::vector<float> LinearLayer::forward(const std::vector<float>& inputs) {
    if (device == CPU) {
        for (size_t i = 0; i < weights.size(); ++i) {
            float sum = biases[i];
            for (size_t j = 0; j < inputs.size(); ++j) {
                sum += weights[i][j] * inputs[j];
            }
            std::cout << "CPU - Pre-activation output neuron " << i << ": " << sum << std::endl;
            outputs[i] = activationFunction(sum); // Sigmoid activation
            std::cout << "CPU - Output neuron " << i << ": " << outputs[i] << std::endl;
        }
    } else if (device == CUDA) {
        matMulCuda(inputs, outputs);
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i] += biases[i];
            std::cout << "CUDA - Pre-activation output neuron " << i << ": " << outputs[i] << std::endl;
            outputs[i] = activationFunction(outputs[i]); // Adding bias and applying activation function
            std::cout << "CUDA - Output neuron " << i << ": " << outputs[i] << std::endl;
        }
    }
    return outputs;
}


void LinearLayer::matMulCuda(const std::vector<float>& inputs, std::vector<float>& outputs) {
    int M = weights.size(); // Number of neurons
    int K = weights[0].size(); // Number of inputs
    int N = 2; // Single input vector for each neuron

    // Replicate inputs for each neuron
    float *a = new float[K * M];
    for (int i = 0; i < M; ++i) {
        std::copy(inputs.begin(), inputs.end(), a + i * K);
    }

    // Flatten weights matrix appropriately
    float *b = new float[K * M]; // Now b should have size K * M
    float *ab = new float[M];

    // Flatten weights matrix in the correct order
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            b[j * M + i] = weights[i][j];
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
    std::cout << "Result ab:" << std::endl;
    for (int i = 0; i < M; ++i) {
        std::cout << " " << ab[i];
    }
    std::cout << std::endl;

    // Assign the result to outputs
    outputs.assign(ab, ab + M);

    // Clean up allocated memory
    delete[] a;
    delete[] b;
    delete[] ab;
}



void LinearLayer::backward(const std::vector<float>& nextLayerDeltas) {
    for (size_t i = 0; i < deltas.size(); ++i) {
        float delta = 0.0f;
        for (size_t j = 0; j < nextLayerDeltas.size(); ++j) {
            delta += nextLayerDeltas[j] * outputs[j];
        }
        deltas[i] = delta * activationFunctionDerivative(outputs[i]);
    }
}

void LinearLayer::updateWeights(float learningRate) {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            weights[i][j] += learningRate * deltas[i] * outputs[j];  // Update weight
        }
        biases[i] += learningRate * deltas[i];  // Update bias
    }
}

std::vector<float> LinearLayer::getOutputs() const {
    return outputs;
}

std::vector<float> LinearLayer::computeDeltas() const {
    return deltas;
}

float LinearLayer::activationFunction(float x) {
    return 1.0 / (1.0 + exp(-x)); // Sigmoid function
}

float LinearLayer::activationFunctionDerivative(float x) {
    float fx = activationFunction(x);
    return fx * (1 - fx); // Derivative of sigmoid
}







