//
// Created by kevin on 06/12/24.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "linear_layer.h"

class NeuralNetwork {
public:
    NeuralNetwork();
    void addLayer(LinearLayer* layer);
    void train(const std::vector<float>& inputs, const std::vector<float>& outputs);
private:
    std::vector<LinearLayer*> layers;
};

#endif // NEURAL_NETWORK_H


