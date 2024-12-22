//
// Created by kevin on 06/12/24.
//

#include <iostream>
#include <vector>

#include "cuda_matmul.h"
#include "dataset_loader.h"
#include "device_type.h"
#include "neural_network.h"
#include "parameters.h"

int main(int argc, char* argv[]) {
    Parameters params(argc, argv);
    // int num_threads = params.getFSamplesNumThreads();
    ParallelImplCpu parallelImplCpu = params.getParallelImplCpu();
    // std::cout << num_threads << std::endl;


    DeviceType device = CUDA;  // or CUDA
    NeuralNetwork nn(device);

    // Add layers
    nn.addLayer(6824, 4, ActivationFunction::ReLU); // Hidden layer with 4 neurons
    nn.addLayer(4, 1, ActivationFunction::Sigmoid); // Output layer with 1 neuron

    if (device == CPU && parallelImplCpu != No) {
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


    // Define the label mapping
    std::unordered_map<std::string, int> label_map = {{"B", 0}, {"M", 1}};
    DatasetLoader loader("../dataset/REJAFADA.data", "", ',', true, true, label_map);
    loader.load();
    loader.normalizeFeatures();
    std::cout << "Features: " << loader.getFeatures()[0].size() << std::endl;
    //std::cout << "Labels: " << loader.getLabels().size() << std::endl;
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
    /*std::vector<std::vector<float>> inputs = {
        {0.2, 0.8},
        {0.5, 0.1},
        {0.9, 0.3},
        {0.7, 0.5}
    };*/

    std::vector<int> labels = loader.getLabels();
    // std::vector<int> labels = {0, 1, 1, 0};

    // Train the network
    nn.train(inputs, labels, 0.1f, 300, 100, No);

    // Test the network
    auto predictions = nn.predict(inputs, parallelImplCpu);
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Input: " << inputs[i][0] << ", " << inputs[i][1] << " -> Predicted: " << predictions[i] << std::endl;
    }

    return 0;
}

/*int main() {
    int N = 2; // Number of rows in matrix a
    int K = 2; // Number of columns in matrix a and rows in matrix b
    int M = 4; // Number of columns in matrix b

    std::vector<std::vector<float>> inputs = {
        {0.2, 0.8},
        {0.5, 0.1}
    };

    std::vector<float> weights = {
        -0.388729, 1.11917, 0.182146, -1.06618,
        -0.991981, -0.73903, -1.3855, -0.222945
    };

    std::vector<std::vector<float>> outputs(N, std::vector<float>(M));

    // Flatten input and weight matrices
    float *a = new float[N * K];
    float *b = new float[K * M];
    float *ab = new float[N * M];

    // Initialize the flattened matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            a[i * K + j] = inputs[i][j];
        }
    }

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < M; ++j) {
            b[i * M + j] = weights[i * M + j];
        }
    }

    // Perform matrix multiplication
    matMul(a, b, ab, M, K, N);

    // Print the result
    std::cout << "Result ab:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << ab[i * M + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] a;
    delete[] b;
    delete[] ab;

    return 0;
}*/



