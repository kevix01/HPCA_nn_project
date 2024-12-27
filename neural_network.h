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

    void setOpenmpThreads(int openmp_threads) {
        this->openmp_threads = openmp_threads;
    }

    void setCudaForwardTileSize(int cuda_forward_tile_size) {
        this->cuda_forward_tile_size = cuda_forward_tile_size;
    }

    void setCudaBackwardBlockSize(int cuda_backward_block_size) {
        this->cuda_backward_block_size = cuda_backward_block_size;
    }

private:
    DeviceType device;
    int openmp_threads;
    int cuda_forward_tile_size;
    int cuda_backward_block_size;
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






