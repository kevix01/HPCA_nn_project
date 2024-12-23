//
// Created by kevin on 06/12/24.
//

#include "linear_layer.h"
#include <cmath>
#include <future>
#include <vector>
#include <iostream>
#include <random>
#include <omp.h>
#include <sys/wait.h>
//#include <cuda_matmul.cu>
#include "cuda_matmul.h"

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
std::vector<std::vector<float>> LinearLayer::forwardCUDA(const std::vector<std::vector<float>>& inputs) {
    inputCache = inputs;
    std::vector<std::vector<float>> output(inputs.size(), std::vector<float>(outputSize));
    /*for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            std::cout << weights[i * inputSize + j] << " ";
        }
        std::cout << std::endl;
    }*/
    matMulCuda(inputs, output);
    /*for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
            // output[i][j] += biases[j];
            // output[i][j] = activate(output[i][j]);
        }
    }*/
    outputCache = output;
    return output;
}

std::vector<std::vector<float>> LinearLayer::forwardCPU(const std::vector<std::vector<float>>& inputs) {
    inputCache = inputs;
    std::vector<std::vector<float>> output(inputs.size(), std::vector<float>(outputSize));
    // print weights
    /*for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            std::cout << weights[i * inputSize + j] << " ";
        }
        std::cout << std::endl;
    }*/
    int sample_id = 0;
    for (auto input : inputs) {
        for (int i = 0; i < outputSize; ++i) {
            float sum = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += input[j] * weights[i * inputSize + j];
                // std::cout << "sum: " << sum << "+=" << input[j] << "*" << weights[i * inputSize + j] << std::endl;
            }
            // std::cout << sum << " ";
            output[sample_id][i] = activate(sum);
        }
        sample_id++;
        // std::cout << std::endl;
    }
    outputCache = output;
    return output;
}

/*std::vector<std::vector<float>> LinearLayer::forwardCPUthreads(const std::vector<std::vector<float>>& inputs, int num_threads) {
    inputCache = inputs;
    std::vector<std::vector<float>> output(inputs.size(), std::vector<float>(outputSize));
    std::mutex mutex;  // Mutex for protecting shared data

    // Function for each thread to process a sample
    auto processSample = [&](int start, int end) {
        for (int sample_id = start; sample_id < end; ++sample_id) {
            for (int i = 0; i < outputSize; ++i) {
                float sum = biases[i];
                for (int j = 0; j < inputSize; ++j) {
                    sum += inputs[sample_id][j] * weights[i * inputSize + j];
                }
                float activated_sum = activate(sum);
                {
                    std::lock_guard<std::mutex> lock(mutex);  // Ensure thread-safe access to output
                    output[sample_id][i] = activated_sum;
                }
            }
        }
    };

    // Determine the number of threads to create
    int real_num_threads = std::min(num_threads, static_cast<int>(inputs.size()));
    int samples_per_thread = inputs.size() / num_threads;
    int remaining_samples = inputs.size() % num_threads;

    // Create and launch threads
    std::vector<std::thread> threads;
    int start = 0;
    for (int i = 0; i < real_num_threads; ++i) {
        int end = start + samples_per_thread + (remaining_samples > 0 ? 1 : 0);
        if (remaining_samples > 0) --remaining_samples;
        threads.push_back(std::thread(processSample, start, end));
        start = end;
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }

    outputCache = output;
    return output;
}*/

std::vector<std::vector<float>> LinearLayer::forwardCPUopenMP(const std::vector<std::vector<float>>& inputs, int f_samples_num_threads, int f_out_neurons_num_threads, int f_in_neurons_num_threads) {
    inputCache = inputs;
    std::vector<std::vector<float>> output(inputs.size(), std::vector<float>(outputSize));

    // Set the number of threads for the outermost parallel region
    // omp_set_num_threads(samples_num_threads);

    int samples_num_threads = std::min(static_cast<int>(inputs.size()), f_samples_num_threads);
    int out_neurons_num_threads = std::min(outputSize, f_out_neurons_num_threads);
    int in_neurons_num_threads = std::min(inputSize, f_in_neurons_num_threads);

    #pragma omp parallel for num_threads(samples_num_threads)
    for (int sample_id = 0; sample_id < inputs.size(); ++sample_id) {
        /*if (omp_get_thread_num() == 0) { // Print only once
            #pragma omp critical
            std::cout << "Total number of threads for outer loop: " << omp_get_num_threads() << std::endl;
        }*/

        const auto& input = inputs[sample_id];

        #pragma omp parallel for num_threads(out_neurons_num_threads)
        for (int i = 0; i < outputSize; ++i) {
            /*if (omp_get_thread_num() == 0) { // Print only once per outer iteration
                #pragma omp critical
                std::cout << "Total number of threads for middle loop: " << omp_get_num_threads() << std::endl;
            }*/

            float sum = biases[i];

            #pragma omp parallel for reduction(+:sum) num_threads(in_neurons_num_threads)
            for (int j = 0; j < inputSize; ++j) {
                /*if (omp_get_thread_num() == 0) { // Print only once per middle iteration
                    #pragma omp critical
                    std::cout << "Total number of threads for inner loop: " << omp_get_num_threads() << std::endl;
                }*/

                sum += input[j] * weights[i * inputSize + j];
            }

            #pragma omp critical
            output[sample_id][i] = activate(sum);
        }
    }

    outputCache = output;
    return output;
}





/*std::vector<std::vector<float>> LinearLayer::forwardCPUprocesses(const std::vector<std::vector<float>>& inputs, int num_processes) {
    inputCache = inputs;
    std::vector<std::vector<float>> output(inputs.size(), std::vector<float>(outputSize));

    // Function for each process to handle a sample
    auto processSample = [&](int start, int end) {
        for (int sample_id = start; sample_id < end; ++sample_id) {
            for (int i = 0; i < outputSize; ++i) {
                float sum = biases[i];
                for (int j = 0; j < inputSize; ++j) {
                    sum += inputs[sample_id][j] * weights[i * inputSize + j];
                }
                output[sample_id][i] = activate(sum);
            }
        }
    };

    // Determine the number of processes to create
    int num_threads = std::min(num_processes, static_cast<int>(inputs.size()));
    int samples_per_thread = inputs.size() / num_threads;
    int remaining_samples = inputs.size() % num_threads;

    // Create futures for each process
    std::vector<std::future<void>> futures;
    int start = 0;
    for (int i = 0; i < num_threads; ++i) {
        int end = start + samples_per_thread + (remaining_samples > 0 ? 1 : 0);
        if (remaining_samples > 0) --remaining_samples;
        futures.push_back(std::async(std::launch::async, processSample, start, end));
        start = end;
    }

    // Wait for all futures to complete
    for (auto& future : futures) {
        future.get();
    }

    outputCache = output;
    return output;
}*/


std::vector<std::vector<float>> LinearLayer::backwardCPUopenMP(const std::vector<std::vector<float>>& grad, float learningRate, int b_out_neurons_num_threads, int b_deltas_num_threads, int b_in_neurons_num_threads) {
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));

    // Determine the number of threads to use to avoid overheading
    int out_neurons_num_threads = std::min(outputSize, b_out_neurons_num_threads);
    int deltas_num_threads = std::min(static_cast<int>(grad.size()), b_deltas_num_threads);
    int in_neurons_num_threads = std::min(inputSize, b_in_neurons_num_threads);

    #pragma omp parallel for num_threads(out_neurons_num_threads) // Parallelize the outer loop
    for (int i = 0; i < outputSize; ++i) {
        /*if (omp_get_thread_num() == 0) { // Print only once per middle iteration
            #pragma omp critical
            std::cout << "Total number of threads for outer loop: " << omp_get_num_threads() << std::endl;
        }*/
        std::vector<float> deltas(grad.size());

        // Parallelize the deltas calculation loop
        #pragma omp parallel for num_threads(deltas_num_threads)
        for (int k = 0; k < grad.size(); ++k) {
            /*if (omp_get_thread_num() == 0) { // Print only once per middle iteration
                #pragma omp critical
                std::cout << "Total number of threads for deltas loop: " << omp_get_num_threads() << std::endl;
            }*/
            deltas[k] = grad[k][i] * activateDerivative(outputCache[k][i]);
        }

        // Average the delta
        float avg_delta = std::accumulate(deltas.begin(), deltas.end(), 0.0f) / deltas.size();

        // Update weights and accumulate gradInput
        #pragma omp parallel for num_threads(in_neurons_num_threads)
        for (int j = 0; j < inputSize; ++j) {
            /*if (omp_get_thread_num() == 0) { // Print only once per middle iteration
                #pragma omp critical
                std::cout << "Total number of threads for inner loop: " << omp_get_num_threads() << std::endl;
            }*/
            float weight_step = 0.0f; // Reset weight_step to zero at each iteration
            #pragma omp parallel for reduction(+:weight_step) num_threads(deltas_num_threads)
            for (int k = 0; k < deltas.size(); ++k) {
                /*if (omp_get_thread_num() == 0) { // Print only once per middle iteration
                    #pragma omp critical
                    std::cout << "Total number of threads for inner loop: " << omp_get_num_threads() << std::endl;
                }*/
                weight_step += deltas[k] * inputCache[k][j];
                #pragma omp atomic
                gradInput[k][j] += deltas[k] * weights[i * inputSize + j];
            }
            weight_step /= deltas.size();
            // #pragma omp atomic
            weights[i * inputSize + j] -= learningRate * weight_step;
        }

        // Update biases
        // #pragma omp atomic
        biases[i] -= learningRate * avg_delta;
    }

    return gradInput;
}





std::vector<std::vector<float>> LinearLayer::backward(const std::vector<std::vector<float>>& grad, float learningRate) {
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));
    for (int i = 0; i < outputSize; ++i) {
        std::vector<float> deltas(grad.size());
        // Calculate deltas for each sample in the batch relative to one of the output neurons
        for (int k = 0; k < grad.size(); ++k) {
            deltas[k] = grad[k][i] * activateDerivative(outputCache[k][i]);
        }

        // Average the delta
        float avg_delta = std::accumulate(deltas.begin(), deltas.end(), 0.0f) / deltas.size();
        // Update weights and accumulate gradInput
        for (int j = 0; j < inputSize; ++j) {
            float weight_step = 0.0f;
            for (int k = 0; k < deltas.size(); ++k) {
                weight_step += deltas[k] * inputCache[k][j];
                gradInput[k][j] += deltas[k] * weights[i * inputSize + j];
            }
            weight_step /= deltas.size();
            //std::cout << "Weight step: " << weight_step << std::endl;
            weights[i * inputSize + j] -= learningRate * weight_step;
        }

        // Update biases
        biases[i] -= learningRate * avg_delta;
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

void LinearLayer::matMulCuda(const std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs) {
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

    // print matrices a and b
    /*std::cout << "Matrix a:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << a[i * K + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Matrix b:" << std::endl;
    for (int i = 0; i < K*M; ++i) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;*/

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
    matMul(a, b, ab, M, K, N, act_type);

    // Print the result
    /*std::cout << "Result ab:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            std::cout << ab[i * M + j] << " ";
        }
        std::cout << std::endl;
    }*/

    // copy results in outputs
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            outputs[i][j] = ab[i * M + j];
        }
    }

    delete[] a;
    delete[] b;
    delete[] ab;
}






