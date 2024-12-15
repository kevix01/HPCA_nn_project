//
// Created by kevin on 06/12/24.
//

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include "device_type.h"

enum class ActivationFunction {
    Sigmoid,
    ReLU
};

class LinearLayer {
public:
    LinearLayer(int inputSize, int outputSize, ActivationFunction activation, DeviceType device, unsigned int seed = 0);
    ~LinearLayer() = default;

    // Forward pass
    std::vector<float> forward(const std::vector<float>& input);

    // Backward pass
    std::vector<float> backward(const std::vector<float>& grad, float learningRate);

    void matMulCuda(const std::vector<float>& inputs, std::vector<float>& outputs);

private:
    int inputSize;
    int outputSize;
    ActivationFunction activation;
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<float> inputCache;
    std::vector<float> outputCache;
    DeviceType device;

    // Activation functions
    float activate(float x);
    float activateDerivative(float x);
};

#endif // LINEAR_LAYER_H





