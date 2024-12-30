//
// Created by kevin on 06/12/24.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

#include "dataset_loader.h"
#include "device_type.h"
#include "forward_cuda.h"
#include "neural_network.h"
#include "parameters.h"
#include "times_printing.h"

std::chrono::duration<double> elapsed_backward = std::chrono::duration<double>::zero();
std::chrono::duration<double> elapsed_forward = std::chrono::duration<double>::zero();
std::chrono::duration<double> elapsed_f_kernel = std::chrono::duration<double>::zero();
std::chrono::duration<double> elapsed_b_kernel_deltas = std::chrono::duration<double>::zero();
std::chrono::duration<double> elapsed_b_kernel_weights = std::chrono::duration<double>::zero();

int main(int argc, char* argv[]) {
    Parameters params(argc, argv);
    // int num_threads = params.getFSamplesNumThreads();
    // std::cout << num_threads << std::endl;


    DeviceType device = params.getDevice();
    int train_batch_size = params.getTrainBatchSize();
    int predict_batch_size = params.getPredictBatchSize();
    int train_epochs = params.getTrainEpochs();
    float learning_rate = params.getLearningRate();
    std::cout << "Device: " << (device == CPU ? "CPU" : "CUDA") << std::endl;
    NeuralNetwork nn(device);

    // Add layers
    nn.addLayer(6824, 50, ActivationFunction::ReLU); // First hidden layer
    nn.addLayer(50, 10, ActivationFunction::ReLU); // Second hidden layer
    nn.addLayer(10, 1, ActivationFunction::Sigmoid); // Output layer with 1 neuron

    // Load needed parameters
    if (device == CPU){
        ParallelImplCpu parallelImplCpu = params.getParallelImplCpu();
        nn.setParallelImplCpu(parallelImplCpu);
        if (parallelImplCpu == OpenMP) {
            nn.setOpenmpThreads(params.getOpenmpThreads());
            std::cout << "Parallelism: OpenMP" << std::endl;
            std::cout << "OpenMP threads: " << params.getOpenmpThreads() << std::endl;
        } else {
            std::cout << "Parallelism: No" << std::endl;
        }
    } else if (device == CUDA) {
        nn.setCudaForwardTileSize(params.getCudaFTileSize());
        nn.setCudaBackwardBlockSize(params.getCudaBBlockSize());
        std::cout << "CUDA forward tile size: " << params.getCudaFTileSize() << std::endl;
        std::cout << "CUDA backward block size: " << params.getCudaBBlockSize() << std::endl;
    }

    std::cout << "########### MODEL PARAMETERS ###########" << std::endl;
    std::cout << "Train batch size: " << train_batch_size << std::endl;
    std::cout << "Train epochs: " << train_epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << "Predict batch size: " << predict_batch_size << std::endl;
    std::cout << "########################################" << std::endl;


    // Define the label mapping
    std::unordered_map<std::string, int> label_map = {{"B", 0}, {"M", 1}};
    DatasetLoader loader("../dataset/REJAFADA.data", "", ',', true, true, label_map);
    std::cout << "Loading dataset..." << std::endl;
    loader.load();
    loader.normalizeFeatures();
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

    // Use only first 2 samples in the dataset
    //inputs = std::vector<std::vector<float>>(inputs.begin(), inputs.begin() + 2);
    // labels = std::vector<int>(labels.begin(), labels.begin() + 2);
    // cut features to the first 100 features
    /*for (auto& feature : inputs) {
        feature = std::vector<float>(feature.begin(), feature.begin() + 4);
    }*/
    std::cout << "############# DATASET INFO #############" << std::endl;
    std::cout << "Features: " << inputs[0].size() << std::endl;
    std::cout << "Number of samples: " << inputs.size() << std::endl;
    std::cout << "Number of labels: " << labels.size() << std::endl;
    std::cout << "########################################" << std::endl;

    //shuffle inputs and labels but keep the correspondence
    std::vector<std::vector<float>> shuffled_inputs;
    std::vector<int> shuffled_labels;
    std::vector<int> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    // shuffle indices with a generator and a specific seed
    int shuffle_seed = 0;
    std::mt19937 g(shuffle_seed);
    std::ranges::shuffle(indices, g);
    for (int i = 0; i < inputs.size(); ++i) {
        shuffled_inputs.push_back(inputs[indices[i]]);
        shuffled_labels.push_back(labels[indices[i]]);
    }

    std::cout << "Training..." << std::endl;
    auto start_train = std::chrono::high_resolution_clock::now();
    // Train the network
    nn.train(shuffled_inputs, shuffled_labels, learning_rate, train_epochs, train_batch_size);
    auto end_train = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_train = end_train - start_train;

    /*std::cout << "################################################## TIMES #################################################" << std::endl;
    // print the time taken to train the network in format hh:mm:ss:ms:us
    std::cout << "Time taken to train the network: " << std::chrono::duration_cast<std::chrono::hours>(elapsed_train).count() << " hours, ";
    std::cout << std::chrono::duration_cast<std::chrono::minutes>(elapsed_train).count() % 60 << " minutes, ";
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(elapsed_train).count() % 60 << " seconds, ";
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_train).count() % 1000 << " milliseconds, ";
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(elapsed_train).count() % 1000 << " microseconds" << std::endl;
    std::cout << "##########################################################################################################" << std::endl;
    */
    std::cout << "############ TRAINING TIMES ############" << std::endl;
    std::cout << "Time taken to train the network:\n";
    printElapsedTime(elapsed_train);
    elapsed_train = std::chrono::duration<double>::zero();
    std::cout << "Time taken to forward the network:\n";
    printElapsedTime(elapsed_forward);
    elapsed_forward = std::chrono::duration<double>::zero();
    std::cout << "Time taken to backward the network:\n";
    printElapsedTime(elapsed_backward);
    elapsed_backward = std::chrono::duration<double>::zero();
    if (device == CUDA) {
        std::cout << "Time taken to compute forward kernel:\n";
        printElapsedTime(elapsed_f_kernel);
        elapsed_f_kernel = std::chrono::duration<double>::zero();
        std::cout << "Time taken to compute backward kernel\nfor deltas and biases updates:\n";
        printElapsedTime(elapsed_b_kernel_deltas);
        elapsed_b_kernel_deltas = std::chrono::duration<double>::zero();
        std::cout << "Time taken to compute backward kernel\nfor grad and weights updates:\n";
        printElapsedTime(elapsed_b_kernel_weights);
        elapsed_b_kernel_weights = std::chrono::duration<double>::zero();
    }

    std::cout << "########################################" << std::endl;

    // Test the network
    int correct;
    std::cout << "Predicting..." << std::endl;
    auto start_predict = std::chrono::high_resolution_clock::now();
    auto predictions = nn.predict(inputs, predict_batch_size);
    auto end_predict = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_predict = end_predict - start_predict;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
        // std::cout << "Input " << i << ": -> Predicted: " << predictions[i] << ", Label: " << labels[i] << std::endl;
    }
    std::cout << "########### PREDICTING TIMES ###########" << std::endl;
    std::cout << "Time taken to predict the samples:\n";
    printElapsedTime(elapsed_predict);
    std::cout << "Time taken to forward the network:\n";
    printElapsedTime(elapsed_forward);
    if (device == CUDA) {
        std::cout << "Time taken to compute forward kernel:\n";
        printElapsedTime(elapsed_f_kernel);
    }
    std::cout << "########################################" << std::endl;
    std::cout << "Predict accuracy: " << static_cast<float>(correct) / inputs.size() << std::endl;
    return 0;
}


