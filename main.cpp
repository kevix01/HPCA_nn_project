//
// Created by kevin on 06/12/24.
//

#include <iostream>
#include <vector>

#include "device_type.h"
#include "neural_network.h"


int main() {
    DeviceType device = CUDA;  // or CUDA
    NeuralNetwork nn(device);

    // Add layers
    nn.addLayer(2, 4, ActivationFunction::ReLU); // Hidden layer with 8 neurons
    nn.addLayer(4, 1, ActivationFunction::Sigmoid); // Output layer with 1 neuron

    // Training data (XOR problem)
    std::vector<std::vector<float>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<int> labels = {0, 1, 1, 0};

    // Train the network
    nn.train(inputs, labels, 0.1f, 650, 4); // Reduce learning rate and increase epochs

    // Test the network
    for (const auto& input : inputs) {
        std::cout << "Input: " << input[0] << ", " << input[1] << " -> Predicted: " << nn.predict(input) << std::endl;
    }

    return 0;
}



