//
// Created by kevin on 06/12/24.
//

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include "device_type.h"

enum class ActivationFunction {
    Sigmoid,
    ReLU
};

class LinearLayer {
public:
    LinearLayer(int inputSize, int outputSize, ActivationFunction activation, unsigned int seed = 0);
    //~LinearLayer() = default;

    // Forward pass
    std::vector<std::vector<float>> forwardCPU(const std::vector<std::vector<float>>& inputs);
    std::vector<std::vector<float>> forwardCPUopenMP(const std::vector<std::vector<float>>& inputs, int f_samples_num_threads, int f_out_neurons_num_threads, int f_in_neurons_num_threads);
    std::vector<std::vector<float>> forwardCPUprocesses(const std::vector<std::vector<float>>& inputs, int num_processes, int output_neurons_num_processes, int input_neurons_num_processes);
    std::vector<std::vector<float>> forwardCUDA(const std::vector<std::vector<float>>& input);

    // Backward pass
    std::vector<std::vector<float>> backward(const std::vector<std::vector<float>>& grad, float learningRate);
    std::vector<std::vector<float>> backwardCPUopenMP(const std::vector<std::vector<float>>& grad, float learningRate, int b_out_neurons_num_threads, int b_deltas_num_threads, int b_in_neurons_num_threads);

    void matMulCuda(const std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs);


private:
    int inputSize;
    int outputSize;
    ActivationFunction activation;
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<std::vector<float>> inputCache;
    std::vector<std::vector<float>> outputCache;

    // Activation functions
    float activate(float x);
    float activateDerivative(float x);
};

#endif // LINEAR_LAYER_H





