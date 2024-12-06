//
// Created by kevin on 06/12/24.
//

#include "linear_layer.h"

LinearLayer::LinearLayer(int numNeurons) {
    for (int i = 0; i < numNeurons; ++i) {
        neurons.emplace_back(Neuron());
    }
}

std::vector<float> LinearLayer::forward(const std::vector<float>& inputs) {
    std::vector<float> outputs;
    for (auto& neuron : neurons) {
        outputs.push_back(neuron.activate(inputs));
    }
    return outputs;
}
