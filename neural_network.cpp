//
// Created by kevin on 06/12/24.
//

#include "neural_network.h"
#include <cmath>
#include <iostream>
#include <algorithm>

#include "times_printing.h"

NeuralNetwork::NeuralNetwork(DeviceType device) {
    this->device = device;
}

void NeuralNetwork::addLayer(int inputSize, int outputSize, ActivationFunction activation) {
    layers.push_back(std::make_unique<LinearLayer>(inputSize, outputSize, activation));
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels,
                          float learningRate, int epochs, int batchSize) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;

        for (size_t i = 0; i < inputs.size(); i += batchSize) {
            int currentBatchSize = std::min(batchSize, static_cast<int>(inputs.size() - i));
            auto inputsBatch = std::vector<std::vector<float>>(inputs.begin() + i, inputs.begin() + i + currentBatchSize);
            auto labelsBatch = std::vector<int>(labels.begin() + i, labels.begin() + i + currentBatchSize);

            auto start_forward = std::chrono::high_resolution_clock::now();
            auto output = forward(inputsBatch);
            auto end_forward = std::chrono::high_resolution_clock::now();
            elapsed_forward += end_forward - start_forward;

            computeLoss(output, labelsBatch, totalLoss);

            auto start_backward = std::chrono::high_resolution_clock::now();
            backward(output, labelsBatch, learningRate);
            auto end_backward = std::chrono::high_resolution_clock::now();
            elapsed_backward += end_backward - start_backward;
        }

        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / inputs.size()
                  << std::endl;
    }
}

std::vector<int> NeuralNetwork::predict(const std::vector<std::vector<float>>& inputs, int batchSize) {
    std::vector<int> predictions;
    //auto elapsed_forward = std::chrono::duration<double>::zero();

    for (size_t i = 0; i < inputs.size(); i += batchSize) {
        int currentBatchSize = std::min(batchSize, static_cast<int>(inputs.size() - i));
        auto inputsBatch = std::vector<std::vector<float>>(inputs.begin() + i, inputs.begin() + i + currentBatchSize);
        auto start_forward = std::chrono::high_resolution_clock::now();
        auto outputs = forward(inputsBatch);
        auto end_forward = std::chrono::high_resolution_clock::now();
        elapsed_forward += end_forward - start_forward;
        for (int j = 0; j < currentBatchSize; ++j) {
            predictions.push_back(outputs[j][0] >= 0.5f ? 1 : 0);
        }
    }
    return predictions;
}

std::vector<std::vector<float>> NeuralNetwork::forward(const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<float>> activations = inputs;
    if (device == CPU) {
        if (parallelImplCpu == No){
            //std::cout << "No parallelism" << std::endl;
            for (auto& layer : layers) {
                activations = layer->forwardCPU(activations);
            }
        }
        else if (parallelImplCpu == OpenMP) {
            //std::cout << "OpenMP parallelism" << std::endl;
            for (auto& layer : layers) {
                activations = layer->forwardCPUopenMP(activations, openmp_threads);
            }
        }
    } else if (device == CUDA) {
        for (auto& layer : layers) {
            activations = layer->forwardCUDA(activations, cuda_forward_tile_size);
        }
    }
    return activations;
}


void NeuralNetwork::backward(const std::vector<std::vector<float>>& output, const std::vector<int>& labels, float learningRate) {
    std::vector<std::vector<float>> grad = {};
    for (size_t i = 0; i < output.size(); ++i) {
        // grad.push_back({output[i][0] - static_cast<float>(labels[i])});
        grad.push_back({2.0f * (output[i][0] - static_cast<float>(labels[i]))});
    }

    // Iterate through the layers in reverse order
    // The returned gradient is the gradient of the loss with respect to the input of the layer
    if (device == CPU) {
        if (parallelImplCpu == No) {
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backwardCPU(grad, learningRate);
            }
        }
        else if (parallelImplCpu == OpenMP) {
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backwardCPUopenMP(grad, learningRate, openmp_threads);
            }
        }
    }
    else if (device == CUDA) {
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backwardCUDA(grad, learningRate, cuda_backward_block_size);
        }
    }
}

void NeuralNetwork::computeLoss(const std::vector<std::vector<float>>& output, std::vector<int> labels, float& totalLoss) {
    std::vector<float> loss(labels.size());
    int sample_id = 0;
    for (auto label : labels) {
        loss[sample_id] = - (label * std::log(output[sample_id][0]) + (1 - label) * std::log(1 - output[sample_id][0]));
        totalLoss += loss[sample_id];
        sample_id++;
    }
}
