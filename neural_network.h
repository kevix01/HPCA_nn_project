//
// Created by kevin on 06/12/24.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>

#include "device_type.h"
#include "linear_layer.h"

class NeuralNetwork {
public:
    explicit NeuralNetwork(DeviceType device);
    ~NeuralNetwork() = default;

    // Add a layer to the network
    void addLayer(int inputSize, int outputSize, ActivationFunction activation);

    // Train the network using mini-batches
    void train(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels,
               float learningRate, int epochs, int batchSize);

    // Predict the output for a given input
    std::vector<int> predict(const std::vector<std::vector<float>>& input);

private:
    DeviceType device;
    std::vector<std::unique_ptr<LinearLayer>> layers;

    // Forward pass
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs);

    // Backward pass
    void backward(const std::vector<std::vector<float>>& output, const std::vector<int>& labels, float learningRate);

    // Compute loss
    void computeLoss(const std::vector<std::vector<float>>& outputs, std::vector<int> labels, float& totalLoss);

    // Compute accuracy
    float computeAccuracy(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels);
};

#endif // NEURAL_NETWORK_H






