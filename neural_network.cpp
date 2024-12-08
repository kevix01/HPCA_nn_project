//
// Created by kevin on 06/12/24.
//

#include "neural_network.h"

#include <iostream>

NeuralNetwork::NeuralNetwork(DeviceType device) : device(device) {}

void NeuralNetwork::addInputLayer(int numInputs) {
    inputData.resize(numInputs);
}

void NeuralNetwork::addLayer(int numInputs, int numNeurons) {
    layers.emplace_back(LinearLayer(numInputs, numNeurons, device));
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& outputs, float learningRate, int epochs, int batchSize) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); i += batchSize) {
            std::vector<std::vector<float>> inputBatch(inputs.begin() + i, inputs.begin() + std::min(i + batchSize, inputs.size()));
            std::vector<std::vector<float>> outputBatch(outputs.begin() + i, outputs.begin() + std::min(i + batchSize, outputs.size()));

            std::vector<std::vector<float>> batchOutputs;
            forward(inputBatch, batchOutputs);
            backward(outputBatch, batchOutputs);
            updateWeights(learningRate, batchSize);
        }
    }
}

void NeuralNetwork::forward(const std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& batchOutputs) {
    batchOutputs = inputs;
    // std::cout << inputs.size() << std::endl;
   /* for (size_t i = 0; i < inputs.size(); ++i) {
        std::cout << "Input: ";
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            std::cout << inputs[i][j] << " ";
        }
    }*/
    int layerIndex = 0;
    for (auto& layer : layers) {
        std::cout << "Layer " << layerIndex++ << std::endl;
        std::vector<std::vector<float>> nextBatchOutputs;
        for (const auto& input : batchOutputs) {
            nextBatchOutputs.push_back(layer.forward(input));
        }
        batchOutputs = nextBatchOutputs;
    }
}

void NeuralNetwork::backward(const std::vector<std::vector<float>>& targets, const std::vector<std::vector<float>>& outputs) {
    for (int i = layers.size() - 1; i >= 0; --i) {
        if (i == layers.size() - 1) {
            for (size_t j = 0; j < outputs.size(); ++j) {
                const auto& output = outputs[j];
                const auto& target = targets[j];
                std::vector<float> errors;
                std::vector<float> deltas;
                for (size_t k = 0; k < output.size(); ++k) {
                    float error = target[k] - output[k];
                    errors.push_back(error);
                    deltas.push_back(error * output[k] * (1 - output[k])); // Simplified example
                }
                layers[i].backward(deltas);
            }
        } else {
            std::vector<float> nextLayerDeltas = layers[i + 1].computeDeltas();
            layers[i].backward(nextLayerDeltas);
        }
    }
}

void NeuralNetwork::updateWeights(float learningRate, int batchSize) {
    for (auto& layer : layers) {
        layer.updateWeights(learningRate / batchSize);
    }
}




