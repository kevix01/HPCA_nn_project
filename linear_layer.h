//
// Created by kevin on 06/12/24.
//

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <chrono>
#include <vector>

extern std::chrono::duration<double> elapsed_backward_cpu;
extern std::chrono::duration<double> elapsed_forward_cpu;

enum class ActivationFunction {
    Sigmoid,
    ReLU
};

class LinearLayer {
public:
    LinearLayer(int inputSize, int outputSize, ActivationFunction activation, unsigned int seed);
    //~LinearLayer() = default;

    // Forward pass
    std::vector<std::vector<float>> forwardCPU(const std::vector<std::vector<float>>& inputs);
    std::vector<std::vector<float>> forwardCPUopenMP(const std::vector<std::vector<float>>& inputs, int total_threads);
    std::vector<std::vector<float>> forwardCPUprocesses(const std::vector<std::vector<float>>& inputs, int num_processes, int output_neurons_num_processes, int input_neurons_num_processes);
    std::vector<std::vector<float>> forwardCUDA(const std::vector<std::vector<float>>& input, int tile_size);

    // Backward pass
    std::vector<std::vector<float>> backwardCPU(const std::vector<std::vector<float>>& grad, float learningRate);
    std::vector<std::vector<float>> backwardCPUopenMP(const std::vector<std::vector<float>>& grad, float learningRate, int total_threads);
    std::vector<std::vector<float>> backwardCUDA(const std::vector<std::vector<float>>& grad, float learningRate, int block_size);

    void performCudaMatMul(const std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs, int tile_size) const;


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





