//
// Created by kevin on 06/12/24.
//

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include "device_type.h"

class LinearLayer {
public:
    LinearLayer(int numInputs, int numNeurons, DeviceType device);
    std::vector<float> forward(const std::vector<float>& inputs);
    void backward(const std::vector<float>& nextLayerDeltas);
    void updateWeights(float learningRate);
    std::vector<float> getOutputs() const;
    std::vector<float> computeDeltas() const;

private:
    std::vector<std::vector<float>> weights; // Weights for the entire layer
    std::vector<float> biases;               // Biases for the entire layer
    std::vector<float> outputs;
    std::vector<float> deltas;
    DeviceType device;

    void matMulCuda(const std::vector<float>& inputs, std::vector<float>& outputs);
    float activationFunction(float x);
    float activationFunctionDerivative(float x);
};

#endif // LINEAR_LAYER_H





