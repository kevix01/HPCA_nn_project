//
// Created by kevin on 06/12/24.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>

#include "parallel_impl_cpu.h"
#include "device_type.h"
#include "linear_layer.h"
#include "parameters.h"


class NeuralNetwork {
public:
    explicit NeuralNetwork(DeviceType device);
    // ~NeuralNetwork() = default;

    // Add a layer to the network
    void addLayer(int inputSize, int outputSize, ActivationFunction activation);

    // Train the network using mini-batches
    void train(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels,
               float learningRate, int epochs, int batchSize, ParallelImplCpu parallelImplCpu);

    // Predict the output for a given input
    std::vector<int> predict(const std::vector<std::vector<float>>& input,
                             ParallelImplCpu parallelImplCpu);

    void setForwardOutNeuronsNumThreads(int forward_out_neurons_num_threads) {
        this->forward_out_neurons_num_threads = forward_out_neurons_num_threads;
    }

    void setForwardInNeuronsNumThreads(int forward_in_neurons_num_threads) {
        this->forward_in_neurons_num_threads = forward_in_neurons_num_threads;
    }

    void setForwardSamplesNumThreads(int forward_samples_num_threads) {
        this->forward_samples_num_threads = forward_samples_num_threads;
    }

    void setBackwardOutNeuronsNumThreads(int backward_out_neurons_num_threads) {
        this->backward_out_neurons_num_threads = backward_out_neurons_num_threads;
    }

    void setBackwardDeltasNumThreads(int backward_deltas_num_threads) {
        this->backward_deltas_num_threads = backward_deltas_num_threads;
    }

    void setBackwardInNeuronsNumThreads(int backward_in_neurons_num_threads) {
        this->backward_in_neurons_num_threads = backward_in_neurons_num_threads;
    }

private:
    DeviceType device;
    int forward_samples_num_threads;
    int forward_out_neurons_num_threads;
    int forward_in_neurons_num_threads;
    int backward_out_neurons_num_threads;
    int backward_deltas_num_threads;
    int backward_in_neurons_num_threads;
    std::vector<std::unique_ptr<LinearLayer>> layers;

    // Forward pass
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs, ParallelImplCpu parallelImplCpu);

    // Backward pass
    void backward(const std::vector<std::vector<float>>& output, const std::vector<int>& labels, float learningRate, ParallelImplCpu parallelImplCpu);

    // Compute loss
    void computeLoss(const std::vector<std::vector<float>>& outputs, std::vector<int> labels, float& totalLoss);

    // Compute accuracy
    float computeAccuracy(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels);
};

#endif // NEURAL_NETWORK_H






