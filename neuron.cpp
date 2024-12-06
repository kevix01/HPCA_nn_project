//
// Created by kevin on 06/12/24.
//

#include "neuron.h"
#include <cmath>

Neuron::Neuron() {
    // Constructor implementation, initializing weights and bias
}

float Neuron::activate(const std::vector<float>& inputs) {
    float sum = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += weights[i] * inputs[i];
    }
    return activationFunction(sum);
}

float Neuron::activationFunction(float x) {
    return 1.0 / (1.0 + exp(-x)); // Sigmoid function
}
