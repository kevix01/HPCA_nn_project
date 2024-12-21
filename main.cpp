//
// Created by kevin on 06/12/24.
//

#include <iostream>
#include <vector>

#include "dataset_loader.h"
#include "device_type.h"
#include "neural_network.h"
#include "parameters.h"

int main(int argc, char* argv[]) {
    Parameters params(argc, argv);
    // int num_threads = params.getFSamplesNumThreads();
    ParallelImplCpu parallelImplCpu = params.getParallelImplCpu();
    // std::cout << num_threads << std::endl;


    DeviceType device = CPU;  // or CUDA
    NeuralNetwork nn(device);

    if (parallelImplCpu != No) {
        nn.setForwardSamplesNumThreads(params.getFSamplesNumThreads());
        nn.setForwardOutNeuronsNumThreads(params.getFOutNeuronsNumThreads());
        nn.setForwardInNeuronsNumThreads(params.getFInNeuronsNumThreads());
        nn.setBackwardOutNeuronsNumThreads(params.getBOutNeuronsNumThreads());
        nn.setBackwardDeltasNumThreads(params.getBDeltasNumThreads());
        nn.setBackwardInNeuronsNumThreads(params.getBInNeuronsNumThreads());
    }
    std::cout << params.getFSamplesNumThreads() << std::endl;
    std::cout << params.getFOutNeuronsNumThreads() << std::endl;
    std::cout << params.getFInNeuronsNumThreads() << std::endl;
    std::cout << params.getBOutNeuronsNumThreads() << std::endl;
    std::cout << params.getBDeltasNumThreads() << std::endl;
    std::cout << params.getBInNeuronsNumThreads() << std::endl;


    // Add layers
    nn.addLayer(6824, 4, ActivationFunction::ReLU); // Hidden layer with 4 neurons
    nn.addLayer(4, 1, ActivationFunction::Sigmoid); // Output layer with 1 neuron

    // Define the label mapping
    std::unordered_map<std::string, int> label_map = {{"B", 0}, {"M", 1}};
    DatasetLoader loader("../dataset/REJAFADA.data", "", ',', true, true, label_map);
    loader.load();
    loader.normalizeFeatures();
    std::cout << "Features: " << loader.getFeatures()[0].size() << std::endl;
    std::cout << "Labels: " << loader.getLabels().size() << std::endl;
    // Print all the features
    /*for (auto feature : loader.getFeatures()) {
        for (auto elem : feature) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }*/
    // Print all the labels
    /*for (auto label : loader.getLabels()) {
        std::cout << label << " ";
    }*/

    // Training data (XOR problem)
    std::vector<std::vector<float>> inputs = loader.getFeatures();
    /*inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };*/

    std::vector<int> labels = loader.getLabels();
    /*labels = {0, 1, 1, 0};*/

    // Train the network
    nn.train(inputs, labels, 0.1f, 350, 100, parallelImplCpu);

    // Test the network
    auto predictions = nn.predict(inputs, parallelImplCpu);
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Input: " << inputs[i][0] << ", " << inputs[i][1] << " -> Predicted: " << predictions[i] << std::endl;
    }

    return 0;
}




