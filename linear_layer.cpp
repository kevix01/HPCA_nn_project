//
// Created by kevin on 06/12/24.
//

#include "linear_layer.h"
#include <cmath>
#include <future>
#include <vector>
#include <random>
#include <omp.h>
#include <sys/wait.h>
#include "forward_cuda.h"
#include "backward_cuda.h"

LinearLayer::LinearLayer(int inputSize, int outputSize, ActivationFunction activation, unsigned int seed)
    : inputSize(inputSize), outputSize(outputSize), activation(activation) {
    std::mt19937 gen(seed);  // Use the provided seed for reproducibility
    std::normal_distribution<> dis(0.0, 1.0 / std::sqrt(inputSize)); // Normal distribution initialization

    weights.resize(inputSize * outputSize);
    biases.resize(outputSize);

    for (auto& weight : weights) {
        weight = dis(gen);
    }

    for (auto& bias : biases) {
        bias = 0.0f; // Initialize biases to zero
    }
}
std::vector<std::vector<float>> LinearLayer::forwardCUDA(const std::vector<std::vector<float>>& inputs, int tile_size) {
    inputCache = inputs;

    std::vector<std::vector<float>> outputs(inputs.size(), std::vector<float>(outputSize));

    int M = outputs[0].size(); // Number of neurons
    int K = inputs[0].size(); // Number of input features
    int N = outputs.size();

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

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            b[j * M + i] = weights[i * K + j];
        }
    }

    // insert in ab elements the bias
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            ab[i * M + j] = biases[j];
        }
    }

    // Select the activation function type
    ActivationFunctionType act_type;
    if (activation == ActivationFunction::ReLU) {
        act_type = RELU;
    } else if (activation == ActivationFunction::Sigmoid) {
        act_type = SIGMOID;
    }
    // Perform matrix multiplication
    forwardMatMul(a, b, ab, M, K, N, act_type, tile_size);

    // copy results in outputs
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            outputs[i][j] = ab[i * M + j];
        }
    }

    delete[] a;
    delete[] b;
    delete[] ab;

    outputCache = outputs;

    return outputs;
}

std::vector<std::vector<float>> LinearLayer::forwardCPU(const std::vector<std::vector<float>>& inputs) {
    inputCache = inputs;

    std::vector<std::vector<float>> outputs(inputs.size(), std::vector<float>(outputSize));

    int sample_id = 0;

    auto start_forwad_compute = std::chrono::high_resolution_clock::now();
    for (auto input : inputs) {
        for (int i = 0; i < outputSize; ++i) {
            float sum = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += input[j] * weights[i * inputSize + j];
            }
            outputs[sample_id][i] = activate(sum);
        }
        sample_id++;
    }
    auto end_forward_compute = std::chrono::high_resolution_clock::now();
    elapsed_forward_cpu += end_forward_compute - start_forwad_compute;

    outputCache = outputs;

    return outputs;
}


std::vector<std::vector<float>> LinearLayer::forwardCPUopenMP(const std::vector<std::vector<float>>& inputs, int total_threads)
{
    inputCache = inputs;
    std::vector<std::vector<float>> outputs(inputs.size(), std::vector<float>(outputSize));

    // Determine optimal thread distribution
    int samples = inputs.size();
    int out_neurons = outputSize;

    // Compute approximate thread distribution
    int samples_num_threads = std::min(samples, total_threads);
    int out_neurons_num_threads = std::max(1, total_threads / samples_num_threads);

    // Adjust if oversubscription occurs
    samples_num_threads = std::min(samples, total_threads / out_neurons_num_threads);
    out_neurons_num_threads = std::min(out_neurons, total_threads / samples_num_threads);

    // Total threads are now bounded by min(samples, total_threads) * min(out_neurons, total_threads)
    int effective_total_threads = samples_num_threads * out_neurons_num_threads;

    auto start_forward_compute = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(effective_total_threads) collapse(2)
    for (int sample_id = 0; sample_id < samples; ++sample_id) {
        for (int i = 0; i < out_neurons; ++i) {
            float sum = biases[i];

            // Compute sum for the current neuron
            for (int j = 0; j < inputSize; ++j) {
                sum += inputs[sample_id][j] * weights[i * inputSize + j];
            }

            // Apply activation and store in output
            outputs[sample_id][i] = activate(sum);
        }
    }
    auto end_forward_compute = std::chrono::high_resolution_clock::now();
    elapsed_forward_cpu += end_forward_compute - start_forward_compute;

    outputCache = outputs;
    return outputs;
}


std::vector<std::vector<float>> LinearLayer::backwardCPUopenMP(const std::vector<std::vector<float>>& grad, float learningRate, int total_threads)
{
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));

    // Determine the number of threads for parallelization
    int num_threads = std::min(total_threads, outputSize * inputSize);

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
                deltas[k] = grad[k][i] * activateDerivative(outputCache[k][i]);
                avg_delta += deltas[k];
                #pragma omp atomic
                gradInput[k][j] += deltas[k] * weights[i * inputSize + j];
            }
            avg_delta /= deltas.size();

            // Update weights
            float weight_step = 0.0f;
            for (int k = 0; k < grad.size(); ++k) {
                weight_step += deltas[k] * inputCache[k][j];
            }
            weight_step /= deltas.size();
            weights[i * inputSize + j] -= learningRate * weight_step;

            // Update biases (only once per `i`, so handled outside the `j` loop)
            if (j == 0) {
                //#pragma omp atomic
                biases[i] -= learningRate * avg_delta;
            }
        }
    }
    auto end_backward_compute = std::chrono::high_resolution_clock::now();
    elapsed_backward_cpu += end_backward_compute - start_backward_compute;

    return gradInput;
}


std::vector<std::vector<float>> LinearLayer::backwardCPU(const std::vector<std::vector<float>>& grad, float learningRate) {
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));

    auto start_backward_compute = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < outputSize; ++i) {
        std::vector<float> deltas(grad.size());
        float avg_delta = 0.0f;

        // Calculate deltas and average delta in a single loop
        for (int k = 0; k < grad.size(); ++k) {
            deltas[k] = grad[k][i] * activateDerivative(outputCache[k][i]);
            //deltas[k] = delta;
            avg_delta += deltas[k];
        }
        avg_delta /= deltas.size();

        // Update biases
        biases[i] -= learningRate * avg_delta;

        // Update weights and accumulate gradInput
        for (int j = 0; j < inputSize; ++j) {
            float weight_step = 0.0f;
            for (int k = 0; k < deltas.size(); ++k) {
                //float delta = deltas[k];
                weight_step += deltas[k] * inputCache[k][j];
                gradInput[k][j] += deltas[k] * weights[i * inputSize + j];
            }
            weight_step /= deltas.size();
            weights[i * inputSize + j] -= learningRate * weight_step;
        }
    }
    auto end_backward_compute = std::chrono::high_resolution_clock::now();
    elapsed_backward_cpu += end_backward_compute - start_backward_compute;

    return gradInput;
}


std::vector<std::vector<float>> LinearLayer::backwardCUDA(const std::vector<std::vector<float>>& grad, float learningRate, int block_size) {
    int batchSize = grad.size();
    int gradFlatSize = batchSize * outputSize;
    int inputFlatSize = batchSize * inputSize;

    // Flatten input vectors for CUDA
    std::vector<float> flatGrad(gradFlatSize);
    std::vector<float> flatOutputCache(gradFlatSize);
    std::vector<float> flatInputCache(inputFlatSize);

    for (int k = 0; k < batchSize; ++k) {
        for (int i = 0; i < outputSize; ++i) {
            flatGrad[k * outputSize + i] = grad[k][i];
            flatOutputCache[k * outputSize + i] = outputCache[k][i];
        }
        for (int j = 0; j < inputSize; ++j) {
            flatInputCache[k * inputSize + j] = inputCache[k][j];
        }
    }

    ActivationFunctionType activationType;
    if (activation == ActivationFunction::ReLU) {
        activationType = RELU;
    } else if (activation == ActivationFunction::Sigmoid) {
        activationType = SIGMOID;
    }

    // Call the CUDA function
    std::vector<float> flatGradInput = backward_cuda(flatGrad, flatOutputCache, flatInputCache, weights, biases,
                                                     outputSize, inputSize, batchSize, learningRate, activationType, block_size);

    // Convert flat gradInput to 2D vector
    std::vector<std::vector<float>> gradInput(batchSize, std::vector<float>(inputSize));
    for (int k = 0; k < batchSize; ++k) {
        for (int j = 0; j < inputSize; ++j) {
            gradInput[k][j] = flatGradInput[k * inputSize + j];
        }
    }
    return gradInput;
}


float LinearLayer::activate(float x) {
    if (activation == ActivationFunction::Sigmoid) {
        return 1.0f / (1.0f + std::exp(-x));
    } else if (activation == ActivationFunction::ReLU) {
        return std::max(0.0f, x);
    }
    return 0.0f;
}

float LinearLayer::activateDerivative(float x) {
    if (activation == ActivationFunction::Sigmoid) {
        float sig = activate(x);
        return sig * (1.0f - sig);
    } else if (activation == ActivationFunction::ReLU) {
        return x > 0.0f ? 1.0f : 0.0f;
    }
    return 0.0f;
}
