//
// Created by kevin on 06/12/24.
//

#include "linear_layer.h"
#include <cmath> // For mathematical functions like std::exp, std::max
#include <future> // For asynchronous operations (not used in this code)
#include <vector> // For std::vector
#include <random> // For random number generation
#include <omp.h> // For OpenMP parallelism
#include <sys/wait.h> // For process control (not used in this code)
#include "forward_cuda.h" // For CUDA forward pass implementation
#include "backward_cuda.h" // For CUDA backward pass implementation

// Constructor for LinearLayer
LinearLayer::LinearLayer(int inputSize, int outputSize, ActivationFunction activation, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), activation(activation) {
    // Initialize random number generator with the provided seed for reproducibility
    std::mt19937 gen(seed);
    // Use a normal distribution with mean 0 and standard deviation 1/sqrt(inputSize) for weight initialization
    std::normal_distribution<> dis(0.0, 1.0 / std::sqrt(inputSize));

    // Resize the weights and biases vectors
    weights.resize(inputSize * outputSize);
    biases.resize(outputSize);

    // Initialize weights using the normal distribution
    for (auto& weight : weights) {
        weight = dis(gen);
    }

    // Initialize biases to zero
    for (auto& bias : biases) {
        bias = 0.0f;
    }
}

// Perform the forward pass using CUDA
std::vector<std::vector<float>> LinearLayer::forwardCUDA(const std::vector<std::vector<float>>& inputs, int tile_size) {
    inputCache = inputs; // Cache the inputs for use in the backward pass

    // Initialize the output matrix
    std::vector<std::vector<float>> outputs(inputs.size(), std::vector<float>(outputSize));

    int M = outputs[0].size(); // Number of neurons
    int K = inputs[0].size(); // Number of input features
    int N = outputs.size(); // Number of samples

    // Flatten input and weight matrices for CUDA
    float *a = new float[N * K]; // Flattened input matrix
    float *b = new float[K * M]; // Flattened weight matrix
    float *ab = new float[N * M]; // Flattened output matrix (initialized with biases)

    // Initialize the flattened input matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            a[i * K + j] = inputs[i][j];
        }
    }

    // Initialize the flattened weight matrix
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            b[j * M + i] = weights[i * K + j];
        }
    }

    // Initialize the flattened output matrix with biases
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            ab[i * M + j] = biases[j];
        }
    }

    // Determine the activation function type for CUDA
    ActivationFunctionType act_type;
    if (activation == ActivationFunction::ReLU) {
        act_type = RELU;
    } else if (activation == ActivationFunction::Sigmoid) {
        act_type = SIGMOID;
    }

    // Perform matrix multiplication and activation using CUDA
    forwardMatMul(a, b, ab, M, K, N, act_type, tile_size);

    // Copy the results back to the output matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            outputs[i][j] = ab[i * M + j];
        }
    }

    // Free the allocated memory
    delete[] a;
    delete[] b;
    delete[] ab;

    outputCache = outputs; // Cache the outputs for use in the backward pass

    return outputs;
}

// Perform the forward pass on the CPU
std::vector<std::vector<float>> LinearLayer::forwardCPU(const std::vector<std::vector<float>>& inputs) {
    inputCache = inputs; // Cache the inputs for use in the backward pass

    // Initialize the output matrix
    std::vector<std::vector<float>> outputs(inputs.size(), std::vector<float>(outputSize));

    int sample_id = 0;

    // Measure the time taken for the forward computation
    auto start_forwad_compute = std::chrono::high_resolution_clock::now();
    for (auto input : inputs) {
        for (int i = 0; i < outputSize; ++i) {
            float sum = biases[i]; // Start with the bias
            for (int j = 0; j < inputSize; ++j) {
                sum += input[j] * weights[i * inputSize + j]; // Compute the weighted sum
            }
            outputs[sample_id][i] = activate(sum); // Apply the activation function
        }
        sample_id++;
    }
    auto end_forward_compute = std::chrono::high_resolution_clock::now();
    elapsed_forward_cpu += end_forward_compute - start_forwad_compute;

    outputCache = outputs; // Cache the outputs for use in the backward pass

    return outputs;
}

// Perform the forward pass on the CPU using OpenMP parallelism
std::vector<std::vector<float>> LinearLayer::forwardCPUopenMP(const std::vector<std::vector<float>>& inputs, int total_threads) {
    inputCache = inputs; // Cache the inputs for use in the backward pass

    // Initialize the output matrix
    std::vector<std::vector<float>> outputs(inputs.size(), std::vector<float>(outputSize));

    // Determine the number of samples and output neurons
    int samples = inputs.size();
    int out_neurons = outputSize;

    // Compute the optimal thread distribution
    int samples_num_threads = std::min(samples, total_threads);
    int out_neurons_num_threads = std::max(1, total_threads / samples_num_threads);

    // Adjust to avoid oversubscription
    samples_num_threads = std::min(samples, total_threads / out_neurons_num_threads);
    out_neurons_num_threads = std::min(out_neurons, total_threads / samples_num_threads);

    // Total threads are now bounded by min(samples, total_threads) * min(out_neurons, total_threads)
    int effective_total_threads = samples_num_threads * out_neurons_num_threads;

    // Measure the time taken for the forward computation
    auto start_forward_compute = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(effective_total_threads) collapse(2)
    for (int sample_id = 0; sample_id < samples; ++sample_id) {
        for (int i = 0; i < out_neurons; ++i) {
            float sum = biases[i]; // Start with the bias
            for (int j = 0; j < inputSize; ++j) {
                sum += inputs[sample_id][j] * weights[i * inputSize + j]; // Compute the weighted sum
            }
            outputs[sample_id][i] = activate(sum); // Apply the activation function
        }
    }
    auto end_forward_compute = std::chrono::high_resolution_clock::now();
    elapsed_forward_cpu += end_forward_compute - start_forward_compute;

    outputCache = outputs; // Cache the outputs for use in the backward pass

    return outputs;
}

// Perform the backward pass on the CPU using OpenMP parallelism
std::vector<std::vector<float>> LinearLayer::backwardCPUopenMP(const std::vector<std::vector<float>>& grad, float learningRate, int total_threads) {
    // Initialize the gradient of the input
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));

    // Determine the number of threads for parallelization
    int num_threads = std::min(total_threads, outputSize * inputSize);

    // Measure the time taken for the backward computation
    auto start_backward_compute = std::chrono::high_resolution_clock::now();
    #pragma omp parallel num_threads(num_threads)
    {
        // Compute the thread ID and map it to the 2D space of outputSize x inputSize
        int thread_id = omp_get_thread_num();
        int total_tasks = outputSize * inputSize;

        for (int task_id = thread_id; task_id < total_tasks; task_id += num_threads) {
            int i = task_id / inputSize; // Row index
            int j = task_id % inputSize; // Column index

            // Compute deltas and accumulate gradients
            std::vector<float> deltas(grad.size());
            float avg_delta = 0.0f;

            for (int k = 0; k < grad.size(); ++k) {
                deltas[k] = grad[k][i] * activateDerivative(outputCache[k][i]); // Compute the delta
                avg_delta += deltas[k]; // Accumulate the average delta
                #pragma omp atomic
                gradInput[k][j] += deltas[k] * weights[i * inputSize + j]; // Accumulate the gradient of the input
            }
            avg_delta /= deltas.size(); // Compute the average delta

            // Update weights
            float weight_step = 0.0f;
            for (int k = 0; k < grad.size(); ++k) {
                weight_step += deltas[k] * inputCache[k][j]; // Compute the weight step
            }
            weight_step /= deltas.size(); // Compute the average weight step
            weights[i * inputSize + j] -= learningRate * weight_step; // Update the weight

            // Update biases (only once per `i`, so handled outside the `j` loop)
            if (j == 0) {
                biases[i] -= learningRate * avg_delta; // Update the bias
            }
        }
    }
    auto end_backward_compute = std::chrono::high_resolution_clock::now();
    elapsed_backward_cpu += end_backward_compute - start_backward_compute;

    return gradInput;
}

// Perform the backward pass on the CPU
std::vector<std::vector<float>> LinearLayer::backwardCPU(const std::vector<std::vector<float>>& grad, float learningRate) {
    // Initialize the gradient of the input
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));

    // Measure the time taken for the backward computation
    auto start_backward_compute = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < outputSize; ++i) {
        std::vector<float> deltas(grad.size());
        float avg_delta = 0.0f;

        // Calculate deltas and average delta in a single loop
        for (int k = 0; k < grad.size(); ++k) {
            deltas[k] = grad[k][i] * activateDerivative(outputCache[k][i]); // Compute the delta
            avg_delta += deltas[k]; // Accumulate the average delta
        }
        avg_delta /= deltas.size(); // Compute the average delta

        // Update biases
        biases[i] -= learningRate * avg_delta; // Update the bias

        // Update weights and accumulate gradInput
        for (int j = 0; j < inputSize; ++j) {
            float weight_step = 0.0f;
            for (int k = 0; k < deltas.size(); ++k) {
                weight_step += deltas[k] * inputCache[k][j]; // Compute the weight step
                gradInput[k][j] += deltas[k] * weights[i * inputSize + j]; // Accumulate the gradient of the input
            }
            weight_step /= deltas.size(); // Compute the average weight step
            weights[i * inputSize + j] -= learningRate * weight_step; // Update the weight
        }
    }
    auto end_backward_compute = std::chrono::high_resolution_clock::now();
    elapsed_backward_cpu += end_backward_compute - start_backward_compute;

    return gradInput;
}

// Perform the backward pass using CUDA
std::vector<std::vector<float>> LinearLayer::backwardCUDA(const std::vector<std::vector<float>>& grad, float learningRate, int block_size) {
    int batchSize = grad.size(); // Number of samples
    int gradFlatSize = batchSize * outputSize; // Size of the flattened gradient
    int inputFlatSize = batchSize * inputSize; // Size of the flattened input

    // Flatten input vectors for CUDA
    std::vector<float> flatGrad(gradFlatSize);
    std::vector<float> flatOutputCache(gradFlatSize);
    std::vector<float> flatInputCache(inputFlatSize);

    // Initialize the flattened gradient and cache matrices
    for (int k = 0; k < batchSize; ++k) {
        for (int i = 0; i < outputSize; ++i) {
            flatGrad[k * outputSize + i] = grad[k][i];
            flatOutputCache[k * outputSize + i] = outputCache[k][i];
        }
        for (int j = 0; j < inputSize; ++j) {
            flatInputCache[k * inputSize + j] = inputCache[k][j];
        }
    }

    // Determine the activation function type for CUDA
    ActivationFunctionType activationType;
    if (activation == ActivationFunction::ReLU) {
        activationType = RELU;
    } else if (activation == ActivationFunction::Sigmoid) {
        activationType = SIGMOID;
    }

    // Call the CUDA function for backward pass
    std::vector<float> flatGradInput = backward_cuda(flatGrad, flatOutputCache, flatInputCache, weights, biases,
                                                     outputSize, inputSize, batchSize, learningRate, activationType, block_size);

    // Convert the flattened gradient of the input back to a 2D vector
    std::vector<std::vector<float>> gradInput(batchSize, std::vector<float>(inputSize));
    for (int k = 0; k < batchSize; ++k) {
        for (int j = 0; j < inputSize; ++j) {
            gradInput[k][j] = flatGradInput[k * inputSize + j];
        }
    }
    return gradInput;
}

// Apply the activation function
float LinearLayer::activate(float x) {
    if (activation == ActivationFunction::Sigmoid) {
        return 1.0f / (1.0f + std::exp(-x)); // Sigmoid activation
    } else if (activation == ActivationFunction::ReLU) {
        return std::max(0.0f, x); // ReLU activation
    }
    return 0.0f; // Default (no activation)
}

// Compute the derivative of the activation function
float LinearLayer::activateDerivative(float x) {
    if (activation == ActivationFunction::Sigmoid) {
        float sig = activate(x);
        return sig * (1.0f - sig); // Derivative of sigmoid
    } else if (activation == ActivationFunction::ReLU) {
        return x > 0.0f ? 1.0f : 0.0f; // Derivative of ReLU
    }
    return 0.0f; // Default (no activation)
}
