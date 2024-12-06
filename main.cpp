//
// Created by kevin on 06/12/24.
//

#include <iostream>
#include "neural_network.h"

int main() {
    NeuralNetwork nn;
    nn.addLayer(new LinearLayer(3));
    nn.addLayer(new LinearLayer(2));

    std::vector<float> inputs = {0.5, 0.3, 0.2};
    std::vector<float> outputs = {1.0, 0.0};

    // nn.train(inputs, outputs);

    std::cout << "Training complete!" << std::endl;
    return 0;
}
