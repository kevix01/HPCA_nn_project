//
// Created by kevin on 06/12/24.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

// Include custom headers for the neural network and related functionality
#include "dataset_loader.h"
#include "device_type.h"
#include "forward_cuda.h"
#include "neural_network.h"
#include "parameters.h"
#include "times_printing.h"

// Global variables to track elapsed time for different operations
std::chrono::duration<double> elapsed_backward = std::chrono::duration<double>::zero(); // Time for backward pass
std::chrono::duration<double> elapsed_forward = std::chrono::duration<double>::zero();  // Time for forward pass
std::chrono::duration<double> elapsed_f_kernel = std::chrono::duration<double>::zero(); // Time for CUDA forward kernel
std::chrono::duration<double> elapsed_b_kernel_deltas = std::chrono::duration<double>::zero(); // Time for CUDA backward kernel (deltas)
std::chrono::duration<double> elapsed_b_kernel_weights = std::chrono::duration<double>::zero(); // Time for CUDA backward kernel (weights)
std::chrono::duration<double> elapsed_forward_cpu = std::chrono::duration<double>::zero(); // Time for CPU forward computation
std::chrono::duration<double> elapsed_backward_cpu = std::chrono::duration<double>::zero(); // Time for CPU backward computation

int main(int argc, char* argv[]) {
    // Parse command-line arguments to get parameters for the neural network
    Parameters params(argc, argv);

    // Get the device type (CPU or CUDA) from the parameters
    DeviceType device = params.getDevice();

    // Get training and prediction parameters from the command line
    int train_batch_size = params.getTrainBatchSize(); // Batch size for training
    int predict_batch_size = params.getPredictBatchSize(); // Batch size for prediction
    int train_epochs = params.getTrainEpochs(); // Number of training epochs
    float learning_rate = params.getLearningRate(); // Learning rate for training

    // Initialize the neural network with the specified device
    NeuralNetwork nn(device);

    // Get the number of neurons for the hidden layers from the parameters
    int neurons_first_hidden_layer = params.getNeuronsFirstHiddenLayer();
    int neurons_second_hidden_layer = params.getNeuronsSecondHiddenLayer();

    int shuffle_seed = params.getTraindataShuffleSeed(); // Seed for shuffling the training data
    int weights_init_seed = params.getWeightsInitSeed(); // Seed for weights initialization

    // Add layers to the neural network
    nn.addLayer(6824, neurons_first_hidden_layer, ActivationFunction::ReLU, weights_init_seed); // First hidden layer with ReLU activation
    if (neurons_second_hidden_layer > 0) {
        nn.addLayer(neurons_first_hidden_layer, neurons_second_hidden_layer, ActivationFunction::ReLU, weights_init_seed); // Second hidden layer with ReLU activation
        nn.addLayer(neurons_second_hidden_layer, 1, ActivationFunction::Sigmoid, weights_init_seed); // Output layer with 1 neuron and Sigmoid activation
    } else { // If no second hidden layer is specified
        nn.addLayer(neurons_first_hidden_layer, 1, ActivationFunction::Sigmoid, weights_init_seed); // Output layer with 1 neuron and Sigmoid activation
    }

    // Print the model architecture
    std::cout << "########## MODEL ARCHITECTURE ##########" << std::endl;
    std::cout << "Input layer: 6824 neurons" << std::endl;
    std::cout << "First hidden layer: " << neurons_first_hidden_layer << " neurons" << std::endl;
    if (neurons_second_hidden_layer > 0) {
        std::cout << "Second hidden layer: " << neurons_second_hidden_layer << " neurons" << std::endl;
    }
    std::cout << "Output layer: 1 neuron" << std::endl;
    std::cout << "ReLU activation for the hidden layers" << std::endl;
    std::cout << "Sigmoid activation for the output layer" << std::endl;
    std::cout << "Threshold for the output layer: 0.5\n( >= 0.5 => class 1, < 0.5 => class 0)" << std::endl;
    std::cout << "########################################" << std::endl;

    // Print the device being used (CPU or CUDA)
    std::cout << "Device: " << (device == CPU ? "CPU" : "CUDA") << std::endl;

    // Configure parallelism for CPU or CUDA
    if (device == CPU) {
        ParallelImplCpu parallelImplCpu = params.getParallelImplCpu(); // Get CPU parallelism implementation
        nn.setParallelImplCpu(parallelImplCpu); // Set parallelism implementation for the neural network
        if (parallelImplCpu == OpenMP) {
            nn.setOpenmpThreads(params.getOpenmpThreads()); // Set the number of OpenMP threads
            std::cout << "Parallelism: OpenMP" << std::endl;
            std::cout << "OpenMP threads: " << params.getOpenmpThreads() << std::endl;
        } else {
            std::cout << "Parallelism: No" << std::endl;
        }
    } else if (device == CUDA) {
        // Set CUDA-specific parameters for forward and backward passes
        nn.setCudaForwardTileSize(params.getCudaFTileSize()); // Set tile size for CUDA forward pass
        nn.setCudaBackwardBlockSize(params.getCudaBBlockSize()); // Set block size for CUDA backward pass
        std::cout << "CUDA forward tile size: " << params.getCudaFTileSize() << std::endl;
        std::cout << "CUDA backward block size: " << params.getCudaBBlockSize() << std::endl;
    }

    // Print model parameters
    std::cout << "########### MODEL PARAMETERS ###########" << std::endl;
    std::cout << "Train batch size: " << train_batch_size << std::endl;
    std::cout << "Train epochs: " << train_epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << "Predict batch size: " << predict_batch_size << std::endl;
    std::cout << "Weights initialization seed: " << weights_init_seed << std::endl;
    std::cout << "Train data shuffle seed: " << shuffle_seed << std::endl;
    std::cout << "########################################" << std::endl;

    // Define the label mapping for the dataset
    std::unordered_map<std::string, int> label_map = {{"B", 0}, {"M", 1}};
    // Load the dataset using the DatasetLoader class
    DatasetLoader loader("../dataset/REJAFADA.data", "", ',', true, true, label_map);
    std::cout << "Loading dataset..." << std::endl;
    loader.load(); // Load the dataset
    loader.normalizeFeatures(); // Normalize the features in the dataset

    // Get the features and labels from the dataset
    std::vector<std::vector<float>> inputs = loader.getFeatures();
    std::vector<int> labels = loader.getLabels();

    // Print dataset information
    std::cout << "############# DATASET INFO #############" << std::endl;
    std::cout << "Features: " << inputs[0].size() << std::endl;
    std::cout << "Number of samples: " << inputs.size() << std::endl;
    std::cout << "Number of labels: " << labels.size() << std::endl;
    std::cout << "########################################" << std::endl;

    // Shuffle the inputs and labels while maintaining their correspondence
    std::vector<std::vector<float>> shuffled_inputs;
    std::vector<int> shuffled_labels;
    std::vector<int> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, 2, ..., n-1
    std::mt19937 g(shuffle_seed); // Random number generator
    std::ranges::shuffle(indices, g); // Shuffle the indices
    for (int i = 0; i < inputs.size(); ++i) {
        shuffled_inputs.push_back(inputs[indices[i]]); // Shuffle inputs
        shuffled_labels.push_back(labels[indices[i]]); // Shuffle labels
    }

    // Train the neural network
    std::cout << "Training..." << std::endl;
    auto start_train = std::chrono::high_resolution_clock::now(); // Start timer for training
    nn.train(shuffled_inputs, shuffled_labels, learning_rate, train_epochs, train_batch_size); // Train the network
    auto end_train = std::chrono::high_resolution_clock::now(); // End timer for training
    std::chrono::duration<double> elapsed_train = end_train - start_train; // Calculate training time

    // Print training times
    std::cout << "############ TRAINING TIMES ############" << std::endl;
    std::cout << "Time taken to train the network:\n";
    printElapsedTime(elapsed_train);

    std::cout << "Time taken to forward the network:\n";
    printElapsedTime(elapsed_forward);
    elapsed_forward = std::chrono::duration<double>::zero(); // Reset forward time
    std::cout << "Time taken to backward the network:\n";
    printElapsedTime(elapsed_backward);
    elapsed_backward = std::chrono::duration<double>::zero(); // Reset backward time
    if (device == CUDA) {
        // Print CUDA-specific kernel times
        std::cout << "Time taken to compute forward kernel:\n";
        printElapsedTime(elapsed_f_kernel);
        elapsed_f_kernel = std::chrono::duration<double>::zero(); // Reset CUDA forward kernel time
        std::cout << "Time taken to compute backward kernel\nfor deltas and biases updates:\n";
        printElapsedTime(elapsed_b_kernel_deltas);
        std::cout << "Time taken to compute backward kernel\nfor grad and weights updates:\n";
        printElapsedTime(elapsed_b_kernel_weights);
    } else {
        // Print CPU-specific computation times
        std::cout << "Time taken to compute forward computation:\n";
        printElapsedTime(elapsed_forward_cpu);
        elapsed_forward_cpu = std::chrono::duration<double>::zero(); // Reset CPU forward time
        std::cout << "Time taken to compute backward computation:\n";
        printElapsedTime(elapsed_backward_cpu);
    }

    std::cout << "########################################" << std::endl;

    // Test the network
    int correct = 0; // Counter for correct predictions
    std::cout << "Predicting..." << std::endl;
    auto start_predict = std::chrono::high_resolution_clock::now(); // Start timer for prediction
    auto predictions = nn.predict(inputs, predict_batch_size); // Make predictions
    auto end_predict = std::chrono::high_resolution_clock::now(); // End timer for prediction
    std::chrono::duration<double> elapsed_predict = end_predict - start_predict; // Calculate prediction time
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct++; // Count correct predictions
        }
    }

    // Print prediction times
    std::cout << "########### PREDICTING TIMES ###########" << std::endl;
    std::cout << "Time taken to predict the samples:\n";
    printElapsedTime(elapsed_predict);
    std::cout << "Time taken to forward the network:\n";
    printElapsedTime(elapsed_forward);
    if (device == CUDA) {
        std::cout << "Time taken to compute forward kernel:\n";
        printElapsedTime(elapsed_f_kernel);
    } else {
        std::cout << "Time taken to compute forward computation:\n";
        printElapsedTime(elapsed_forward_cpu);
    }
    std::cout << "########################################" << std::endl;

    // Print prediction accuracy
    std::cout << "Predict accuracy: " << static_cast<float>(correct) / inputs.size() << std::endl;
    return 0;
}
