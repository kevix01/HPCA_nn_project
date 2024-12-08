//
// Created by kevin on 06/12/24.
//

#include <iostream>
#include "neural_network.h"

int main() {
    DeviceType device = CUDA;  // or CPU

    NeuralNetwork nn(device);
    nn.addInputLayer(3);         // Example input layer with 3 inputs
    nn.addLayer(3, 2);           // First hidden layer with 3 inputs and 2 neurons
    nn.addLayer(2, 1);           // Output layer with 2 inputs and 1 neuron

    std::vector<std::vector<float>> inputs = {{0.5, 0.3, 0.2}, {0.6, 0.4, 0.1}};
    std::vector<std::vector<float>> outputs = {{1.0}, {0.0}};

    float learningRate = 0.01;
    int epochs = 1;
    int batchSize = 2;

    nn.train(inputs, outputs, learningRate, epochs, batchSize);

    std::cout << "Training complete!" << std::endl;
    return 0;
}



