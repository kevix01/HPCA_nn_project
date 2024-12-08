//
// Created by kevin on 06/12/24.
//

#include "neuron.h"
#include <cmath>

Neuron::Neuron(DeviceType device, int numInputs) : device(device), bias(0.0f), output(0.0f), delta(0.0f) {
    // Initialization if needed
}

float Neuron::activate(const std::vector<float>& inputs) {
    float sum = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        sum += inputs[i];  // Simplified activation assuming direct input
    }
    output = activationFunction(sum);
    return output;
}

float Neuron::activationFunction(float x) {
    return 1.0 / (1.0 + exp(-x)); // Sigmoid function
}

float Neuron::activationFunctionDerivative(float x) {
    float fx = activationFunction(x);
    return fx * (1 - fx); // Derivative of sigmoid
}

float Neuron::getOutput() const {
    return output;
}

void Neuron::setDelta(float delta) {
    this->delta = delta;
}

float Neuron::getDelta() const {
    return delta;
}




