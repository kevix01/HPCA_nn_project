//
// Created by kevin on 06/12/24.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "linear_layer.h"
#include "device_type.h"

class NeuralNetwork {
public:
    NeuralNetwork(DeviceType device);
    void addInputLayer(int numInputs);
    void addLayer(int numInputs, int numNeurons);
    void train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& outputs, float learningRate, int epochs, int batchSize);
private:
    std::vector<LinearLayer> layers;
    DeviceType device;
    std::vector<float> inputData; // to store input data
    void forward(const std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& batchOutputs);
    void backward(const std::vector<std::vector<float>>& targets, const std::vector<std::vector<float>>& outputs);
    void updateWeights(float learningRate, int batchSize);
};

#endif // NEURAL_NETWORK_H






