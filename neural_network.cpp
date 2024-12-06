//
// Created by kevin on 06/12/24.
//

#include "neural_network.h"

NeuralNetwork::NeuralNetwork() {
    // Constructor implementation
}

void NeuralNetwork::addLayer(LinearLayer* layer) {
    layers.push_back(layer);
}

void NeuralNetwork::train(const std::vector<float>& inputs, const std::vector<float>& outputs) {
    // Training implementation
}
