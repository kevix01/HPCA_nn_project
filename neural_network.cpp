//
// Created by kevin on 06/12/24.
//

#include "neural_network.h"
#include <cmath> // For mathematical functions like std::log
#include <iostream>
#include <algorithm> // For std::min

// Constructor for NeuralNetwork
NeuralNetwork::NeuralNetwork(DeviceType device) {
    this->device = device; // Initialize the device type (CPU or CUDA)
}

// Add a layer to the neural network
void NeuralNetwork::addLayer(int inputSize, int outputSize, ActivationFunction activation, int weightsInitSeed) {
    // Create a new LinearLayer and add it to the layers vector
    layers.push_back(std::make_unique<LinearLayer>(inputSize, outputSize, activation, weightsInitSeed));
}

// Train the neural network
void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels,
                          float learningRate, int epochs, int batchSize) {
    // Iterate over the specified number of epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f; // Accumulate the total loss for the epoch

        // Process the dataset in batches
        for (size_t i = 0; i < inputs.size(); i += batchSize) {
            // Determine the size of the current batch
            int currentBatchSize = std::min(batchSize, static_cast<int>(inputs.size() - i));
            // Extract the current batch of inputs and labels
            auto inputsBatch = std::vector<std::vector<float>>(inputs.begin() + i, inputs.begin() + i + currentBatchSize);
            auto labelsBatch = std::vector<int>(labels.begin() + i, labels.begin() + i + currentBatchSize);

            // Perform the forward pass and measure the time taken
            auto start_forward = std::chrono::high_resolution_clock::now();
            auto output = forward(inputsBatch);
            auto end_forward = std::chrono::high_resolution_clock::now();
            elapsed_forward += end_forward - start_forward;

            // Compute the loss for the current batch
            computeLoss(output, labelsBatch, totalLoss);

            // Perform the backward pass and measure the time taken
            auto start_backward = std::chrono::high_resolution_clock::now();
            backward(output, labelsBatch, learningRate);
            auto end_backward = std::chrono::high_resolution_clock::now();
            elapsed_backward += end_backward - start_backward;
        }

        // Print the average loss for the epoch
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / inputs.size()
                  << std::endl;
    }
}

// Predict the output for a set of inputs
std::vector<int> NeuralNetwork::predict(const std::vector<std::vector<float>>& inputs, int batchSize) {
    std::vector<int> predictions; // Vector to store the predictions

    // Process the dataset in batches
    for (size_t i = 0; i < inputs.size(); i += batchSize) {
        // Determine the size of the current batch
        int currentBatchSize = std::min(batchSize, static_cast<int>(inputs.size() - i));
        // Extract the current batch of inputs
        auto inputsBatch = std::vector<std::vector<float>>(inputs.begin() + i, inputs.begin() + i + currentBatchSize);
        // Perform the forward pass and measure the time taken
        auto start_forward = std::chrono::high_resolution_clock::now();
        auto outputs = forward(inputsBatch);
        auto end_forward = std::chrono::high_resolution_clock::now();
        elapsed_forward += end_forward - start_forward;
        // Convert the output probabilities to binary predictions (0 or 1)
        for (int j = 0; j < currentBatchSize; ++j) {
            predictions.push_back(outputs[j][0] >= 0.5f ? 1 : 0);
        }
    }
    return predictions;
}

// Perform the forward pass through the network
std::vector<std::vector<float>> NeuralNetwork::forward(const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> activations = inputs; // Start with the input data

    // Perform the forward pass based on the device type
    if (device == CPU) {
        if (parallelImplCpu == No) {
            // Use the CPU implementation without parallelism
            for (auto& layer : layers) {
                activations = layer->forwardCPU(activations);
            }
        }
        else if (parallelImplCpu == OpenMP) {
            // Use the CPU implementation with OpenMP parallelism
            for (auto& layer : layers) {
                activations = layer->forwardCPUopenMP(activations, openmp_threads);
            }
        }
    } else if (device == CUDA) {
        // Use the CUDA implementation
        for (auto& layer : layers) {
            activations = layer->forwardCUDA(activations, cuda_forward_tile_size);
        }
    }
    return activations; // Return the final output of the network
}

// Perform the backward pass through the network
void NeuralNetwork::backward(const std::vector<std::vector<float>>& output, const std::vector<int>& labels, float learningRate) {
    // Compute the gradient of the loss with respect to the output
    std::vector<std::vector<float>> grad = {};
    for (size_t i = 0; i < output.size(); ++i) {
        grad.push_back({2.0f * (output[i][0] - static_cast<float>(labels[i]))});
    }

    // Iterate through the layers in reverse order
    if (device == CPU) {
        if (parallelImplCpu == No) {
            // Use the CPU implementation without parallelism
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backwardCPU(grad, learningRate);
            }
        }
        else if (parallelImplCpu == OpenMP) {
            // Use the CPU implementation with OpenMP parallelism
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backwardCPUopenMP(grad, learningRate, openmp_threads);
            }
        }
    }
    else if (device == CUDA) {
        // Use the CUDA implementation
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backwardCUDA(grad, learningRate, cuda_backward_block_size);
        }
    }
}

// Compute the loss for a batch of outputs and labels
void NeuralNetwork::computeLoss(const std::vector<std::vector<float>>& output, std::vector<int> labels, float& totalLoss) {
    std::vector<float> loss(labels.size()); // Vector to store the loss for each sample
    int sample_id = 0;
    for (auto label : labels) {
        // Compute the binary cross-entropy loss for each sample
        loss[sample_id] = - (label * std::log(output[sample_id][0]) + (1 - label) * std::log(1 - output[sample_id][0]));
        totalLoss += loss[sample_id]; // Accumulate the total loss
        sample_id++;
    }
}
