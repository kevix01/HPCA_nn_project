//
// Created by kevin on 06/12/24.
//

#include "neural_network.h"
#include <cmath>
#include <iostream>
#include <algorithm>

NeuralNetwork::NeuralNetwork(DeviceType device) {
    this->device = device;
}

void NeuralNetwork::addLayer(int inputSize, int outputSize, ActivationFunction activation) {
    layers.push_back(std::make_unique<LinearLayer>(inputSize, outputSize, activation, device));
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels,
                          float learningRate, int epochs, int batchSize) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < inputs.size(); i += batchSize) {
            int currentBatchSize = std::min(batchSize, static_cast<int>(inputs.size() - i));
            for (int j = 0; j < currentBatchSize; ++j) {
                auto input = inputs[i + j];
                auto output = forward(input);
                int label = labels[i + j];

                totalLoss += computeLoss(output, label);
                correct += (output[0] >= 0.5f) == label;

                backward(output, label, learningRate);
            }
        }

        float accuracy = static_cast<float>(correct) / inputs.size();
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / inputs.size()
                  << ", Accuracy: " << accuracy << std::endl;
    }
}

int NeuralNetwork::predict(const std::vector<float>& input) {
    auto output = forward(input);
    return output[0] >= 0.5f ? 1 : 0;
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    std::vector<float> activation = input;
    for (auto& layer : layers) {
        activation = layer->forward(activation);
    }
    return activation;
}

void NeuralNetwork::backward(const std::vector<float>& output, int label, float learningRate) {
    std::vector<float> grad = {output[0] - static_cast<float>(label)}; // Binary cross-entropy derivative

    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad, learningRate);
    }
}

float NeuralNetwork::computeLoss(const std::vector<float>& output, int label) {
    float loss = - (label * std::log(output[0]) + (1 - label) * std::log(1 - output[0]));
    return loss;
}

float NeuralNetwork::computeAccuracy(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        correct += (predict(inputs[i]) == labels[i]);
    }
    return static_cast<float>(correct) / inputs.size();
}





