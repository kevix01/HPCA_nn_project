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
    // print input
    /*std::cout << "Input: " << std::endl;
    for (int i = 0; i < inputs.size(); ++i) {
        for (int j = 0; j < inputs[i].size(); ++j) {
            std::cout << inputs[i][j] << " ";
        }
        std::cout << std::endl;
    }*/
    std::vector<std::vector<float>> output(inputs.size(), std::vector<float>(outputSize));
    /*std::cout << "Weights: " << std::endl;
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            std::cout << weights[i * inputSize + j] << " ";
        }
        std::cout << std::endl;
    }*/
    // print biases
    /*std::cout << "Biases: " << std::endl;
    for (int i = 0; i < outputSize; ++i) {
        std::cout << biases[i] << " ";
    }
    std::cout << std::endl;*/
    matMulCuda(inputs, output, tile_size);
    /*for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[i].size(); ++j) {
            // output[i][j] += biases[j];
            // output[i][j] = activate(output[i][j]);
        }
    }*/
    outputCache = output;
    // print output
    /*std::cout << "Output: " << std::endl;
    for (int i = 0; i < output.size(); ++i) {
        for (int j = 0; j < output[i].size(); ++j) {
            std::cout << output[i][j] << " ";
        }
        std::cout << std::endl;
    }*/
    return output;
}

std::vector<std::vector<float>> LinearLayer::forwardCPU(const std::vector<std::vector<float>>& inputs) {
    inputCache = inputs;
    // print input
    std::cout << "Input: " << std::endl;
    for (int i = 0; i < inputs.size(); ++i) {
        for (int j = 0; j < inputs[i].size(); ++j) {
            std::cout << inputs[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::vector<std::vector<float>> output(inputs.size(), std::vector<float>(outputSize));
    // print weights
    std::cout << "Weights: " << std::endl;
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            std::cout << weights[i * inputSize + j] << " ";
        }
        std::cout << std::endl;
    }
    // print biases
    std::cout << "Biases: " << std::endl;
    for (int i = 0; i < outputSize; ++i) {
        std::cout << biases[i] << " ";
    }
    std::cout << std::endl;
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
    // print output
    std::cout << "Output: " << std::endl;
    for (int i = 0; i < output.size(); ++i) {
        for (int j = 0; j < output[i].size(); ++j) {
            std::cout << output[i][j] << " ";
        }
        std::cout << std::endl;
    }
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

/*std::vector<std::vector<float>> LinearLayer::forwardCPUopenMP(const std::vector<std::vector<float>>& inputs, int f_samples_num_threads, int f_out_neurons_num_threads) {
    inputCache = inputs;
    std::vector<std::vector<float>> output(inputs.size(), std::vector<float>(outputSize));

    // Set the number of threads for the outermost parallel region
    // omp_set_num_threads(samples_num_threads);

    int samples_num_threads = std::min(static_cast<int>(inputs.size()), f_samples_num_threads);
    int out_neurons_num_threads = std::min(outputSize, f_out_neurons_num_threads);
    // int in_neurons_num_threads = std::min(inputSize, f_in_neurons_num_threads);

    #pragma omp parallel for num_threads(samples_num_threads)
    for (int sample_id = 0; sample_id < inputs.size(); ++sample_id) {
        /*if (omp_get_thread_num() == 0) { // Print only once
            #pragma omp critical
            std::cout << "Total number of threads for outer loop: " << omp_get_num_threads() << std::endl;
        }

        const auto& input = inputs[sample_id];

        #pragma omp parallel for num_threads(out_neurons_num_threads)
        for (int i = 0; i < outputSize; ++i) {
            /*if (omp_get_thread_num() == 0) { // Print only once per outer iteration
                #pragma omp critical
                std::cout << "Total number of threads for middle loop: " << omp_get_num_threads() << std::endl;
            }

            float sum = biases[i];

            // #pragma omp parallel for reduction(+:sum) num_threads(2)
            for (int j = 0; j < inputSize; ++j) {
                /*if (omp_get_thread_num() == 0) { // Print only once per middle iteration
                    #pragma omp critical
                    std::cout << "Total number of threads for inner loop: " << omp_get_num_threads() << std::endl;
                }

                sum += input[j] * weights[i * inputSize + j];
            }

            #pragma omp critical
            output[sample_id][i] = activate(sum);
        }
    }

    outputCache = output;
    return output;
}*/

std::vector<std::vector<float>> LinearLayer::forwardCPUopenMP(const std::vector<std::vector<float>>& inputs, int total_threads)
{
    inputCache = inputs;
    std::vector<std::vector<float>> output(inputs.size(), std::vector<float>(outputSize));

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

    #pragma omp parallel for num_threads(effective_total_threads) collapse(2)
    for (int sample_id = 0; sample_id < samples; ++sample_id) {
        for (int i = 0; i < out_neurons; ++i) {
            float sum = biases[i];

            // Compute sum for the current neuron
            for (int j = 0; j < inputSize; ++j) {
                sum += inputs[sample_id][j] * weights[i * inputSize + j];
            }

            // Apply activation and store in output
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


/*std::vector<std::vector<float>> LinearLayer::backwardCPUopenMP(const std::vector<std::vector<float>>& grad, float learningRate, int b_out_neurons_num_threads, int b_in_neurons_num_threads) {
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));

    // Determine the number of threads to use to avoid overheading
    int out_neurons_num_threads = std::min(outputSize, b_out_neurons_num_threads);
    // int deltas_num_threads = std::min(static_cast<int>(grad.size()), b_deltas_num_threads);
    int in_neurons_num_threads = std::min(inputSize, b_in_neurons_num_threads);

    #pragma omp parallel for num_threads(out_neurons_num_threads) // Parallelize the outer loop
    for (int i = 0; i < outputSize; ++i) {
        /*if (omp_get_thread_num() == 0) { // Print only once per middle iteration
            #pragma omp critical
            std::cout << "Total number of threads for outer loop: " << omp_get_num_threads() << std::endl;
        }
        std::vector<float> deltas(grad.size());
        float avg_delta = 0.0f;

        // Parallelize the deltas calculation loop
        // #pragma omp parallel for reduction(+:avg_delta) num_threads(12)
        for (int k = 0; k < grad.size(); ++k) {
            deltas[k] = grad[k][i] * activateDerivative(outputCache[k][i]);
            //deltas[k] = delta;
            avg_delta += deltas[k];
        }
        avg_delta /= deltas.size();

        // Update weights and accumulate gradInput
        #pragma omp parallel for num_threads(in_neurons_num_threads)
        for (int j = 0; j < inputSize; ++j) {
            /*if (omp_get_thread_num() == 0) { // Print only once per middle iteration
                #pragma omp critical
                std::cout << "Total number of threads for inner loop: " << omp_get_num_threads() << std::endl;
            }
            float weight_step = 0.0f; // Reset weight_step to zero at each iteration
            // #pragma omp parallel for reduction(+:weight_step) num_threads(deltas_num_threads)
            for (int k = 0; k < deltas.size(); ++k) {
                /*if (omp_get_thread_num() == 0) { // Print only once per middle iteration
                    #pragma omp critical
                    std::cout << "Total number of threads for inner loop: " << omp_get_num_threads() << std::endl;
                }
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
}*/

std::vector<std::vector<float>> LinearLayer::backwardCPUopenMP(const std::vector<std::vector<float>>& grad, float learningRate, int total_threads)
{
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));

    // Determine the number of threads for parallelization
    int num_threads = std::min(total_threads, outputSize * inputSize);

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

    return gradInput;
}


// NO FLATTENED DELTAS AND NOT DELTAS SAVING -> LESS MEMORY USAGE
/*std::vector<std::vector<float>> LinearLayer::backward(const std::vector<std::vector<float>>& grad, float learningRate) {
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));

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

        // Update biases
        biases[i] -= learningRate * avg_delta;
    }

    return gradInput;
}*/

std::vector<std::vector<float>> LinearLayer::backward(const std::vector<std::vector<float>>& grad, float learningRate) {
    std::vector<std::vector<float>> gradInput(grad.size(), std::vector<float>(inputSize, 0.0f));
    std::vector<float> flattenedDeltas(grad.size() * outputSize, 0.0f); // Flattened deltas storage

    /*if (outputSize == 4) {
        // print all grad given
        for (int i = 0; i < grad.size(); i++) {
            for (int j = 0; j < outputSize; j++) {
                std::cout << "Grad " << i << " " << j << ": " << grad[i][j] << std::endl;
            }
        }
    }*/

    for (int i = 0; i < outputSize; ++i) {
        float avg_delta = 0.0f;

        // Calculate deltas and average delta in a single loop
        for (int k = 0; k < grad.size(); ++k) {
            float delta = grad[k][i] * activateDerivative(outputCache[k][i]);
            avg_delta += delta;
            flattenedDeltas[k * outputSize + i] = delta; // Store in flattened order
        }
        avg_delta /= grad.size();
        //std::cout << "Avg delta for neuron " << i << ": " << avg_delta << std::endl;

        // Update weights and accumulate gradInput
        for (int j = 0; j < inputSize; ++j) {
            float weight_step = 0.0f;
            for (int k = 0; k < grad.size(); ++k) {
                weight_step += flattenedDeltas[k * outputSize + i] * inputCache[k][j];
                //if (inputSize <=4)
                    //std::cout << "Partial weight step for neuron with delta index " << k*outputSize + i << " and input value " << inputCache[k][j] << ", sample " << k << " and input " << j << ": " << flattenedDeltas[k * outputSize + i] * inputCache[k][j] << std::endl;
                gradInput[k][j] += flattenedDeltas[k * outputSize + i] * weights[i * inputSize + j];
            }
            //if (inputSize <=4) {
            // print all gradInput values
            /*for (int k = 0; k < grad.size(); ++k) {
                std::cout << "GradInput for sample " << k << " and input " << j << ": " << gradInput[k][j] << std::endl;
            }*/
            //}
            // std::cout << grad.size();
            weight_step /= grad.size();
            //if (inputSize <=4)
            // std::cout << "Weight step update for neuron " << i << " and input " << j << ": " << weight_step << std::endl;
            weights[i * inputSize + j] -= learningRate * weight_step;
            //if (inputSize > 4)
                //std::cout << "Weight update for neuron " << i << " and input " << j << ": " << weights[i * inputSize + j] << std::endl;
        }

        // Update biases
        biases[i] -= learningRate * avg_delta;
    }
    /*if (inputSize > 4) {
        std::cout << "Biases" << std::endl;
        // Print the biases
        for (int i = 0; i<biases.size(); ++i) {
            std::cout << biases[i] << " ";
        }
        std::cout << std::endl;
    }*/
    // Print flattened deltas for debugging
    /*std::cout << "Flattened deltas:\n";
    for (size_t idx = 0; idx < flattenedDeltas.size(); ++idx) {
        std::cout << flattenedDeltas[idx] << " ";
        if ((idx + 1) % outputSize == 0) std::cout << "\n"; // New line after each batch
    }
    std::cout << std::endl;*/

    /*for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            std::cout << "Weight " << i << " " << j << ": " << weights[i * inputSize + j] << std::endl;
        }
    }*/


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

    /*std::cout << "Biases" << std::endl;
    // Print the biases
    for (int i = 0; i<biases.size(); ++i) {
        std::cout << biases[i] << " ";
    }
    std::cout << std::endl;*/

    return gradInput;
}


/*std::vector<std::vector<float>> LinearLayer::backward(const std::vector<std::vector<float>>& grad, float learningRate) {
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
}*/


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

void LinearLayer::matMulCuda(const std::vector<std::vector<float>>& inputs, std::vector<std::vector<float>>& outputs, int tile_size) {
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
    matMul(a, b, ab, M, K, N, act_type, tile_size);

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






