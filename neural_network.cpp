//
// Created by kevin on 06/12/24.
//

#include "neural_network.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <omp.h>

NeuralNetwork::NeuralNetwork(DeviceType device): forward_in_neurons_num_threads(NULL), forward_out_neurons_num_threads(NULL) {
    this->device = device;
}

void NeuralNetwork::addLayer(int inputSize, int outputSize, ActivationFunction activation) {
    layers.push_back(std::make_unique<LinearLayer>(inputSize, outputSize, activation));
}

void NeuralNetwork::train(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels,
                          float learningRate, int epochs, int batchSize, ParallelImplCpu parallelImplCpu) {
    if (device == CPU && parallelImplCpu == OpenMP) {
        // Enable nested parallelism globally
        omp_set_nested(1);
    }
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < inputs.size(); i += batchSize) {
            int currentBatchSize = std::min(batchSize, static_cast<int>(inputs.size() - i));
            auto inputsBatch = std::vector<std::vector<float>>(inputs.begin() + i, inputs.begin() + i + currentBatchSize);
            auto labelsBatch = std::vector<int>(labels.begin() + i, labels.begin() + i + currentBatchSize);

            auto output = forward(inputsBatch, parallelImplCpu);
            computeLoss(output, labelsBatch, totalLoss);

            for (int j = 0; j < currentBatchSize; ++j) {
                correct += (output[j][0] >= 0.5f) == labelsBatch[j];
                // std::cout << "Output: " << output[j][0] << std::endl;
            }

            backward(output, labelsBatch, learningRate, parallelImplCpu);

            /*for (int j = 0; j < currentBatchSize; ++j) {
                auto input = inputs[i + j];
                auto output = forward(input);
                int label = labels[i + j];

                computeLoss(output, labelsBatch, totalLoss);
                correct += (output[0] >= 0.5f) == label;

                backward(output, label, learningRate);
            }*/
        }

        float accuracy = static_cast<float>(correct) / inputs.size();
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / inputs.size()
                  << ", Accuracy: " << accuracy << std::endl;
    }
}

std::vector<int> NeuralNetwork::predict(const std::vector<std::vector<float>>& input, ParallelImplCpu parallelImplCpu) {
    if (device == CPU && parallelImplCpu == OpenMP) {
        // Enable nested parallelism globally
        omp_set_nested(1);
    }
    auto output = forward(input, parallelImplCpu);
    std::vector<int> predictions;
    for (auto& out : output) {
        // std::cout << "Output: " << out[0] << std::endl;
        predictions.push_back(out[0] >= 0.5f ? 1 : 0);
    }
    return predictions;
}

std::vector<std::vector<float>> NeuralNetwork::forward(const std::vector<std::vector<float>>& inputs, ParallelImplCpu parallelImplCpu) {
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
                activations = layer->forwardCPUopenMP(activations, forward_samples_num_threads, forward_out_neurons_num_threads);
            }
        } else if (parallelImplCpu == processes) {
            for (auto& layer : layers) {
               // activations = layer->forwardCPUprocesses(activations, num_processes, 40, 40);
            }
        }
        //return activations;
    } else if (device == CUDA) {
        for (auto& layer : layers) {
            activations = layer->forwardCUDA(activations);
        }
    }
    return activations;
}


void NeuralNetwork::backward(const std::vector<std::vector<float>>& output, const std::vector<int>& labels, float learningRate, ParallelImplCpu parallelImplCpu) {
    std::vector<std::vector<float>> grad = {};
    for (size_t i = 0; i < output.size(); ++i) {
        grad.push_back({output[i][0] - static_cast<float>(labels[i])});
    }
    /*std::cout << "Grad elements: " << grad.size() << std::endl;
    std::cout << "Elements in grad: " ;
    for (auto grad_elem : grad) {
        for (auto elem : grad_elem) {
            std::cout << elem << " ";
        }
    }
    std::cout << std::endl;*/
    // compute the average gradient
    /*std::vector<float> avg_grad(grad[0].size(), 0.0f);
    for (size_t i = 0; i < grad.size(); ++i) {
        for (size_t j = 0; j < grad[i].size(); ++j) {
            avg_grad[j] += grad[i][j];
        }
    }
    for (size_t i = 0; i < avg_grad.size(); ++i) {
        avg_grad[i] /= grad.size();
    }*/
    // Iterate through the layers in reverse order
    // The returned gradient is the gradient of the loss with respect to the input of the layer
    if (device == CPU) {
        if (parallelImplCpu == No) {
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backward(grad, learningRate);
                //std::cout << "New elements in grad: " ;
                /*for (auto grad_elem : grad) {
                    for (auto elem : grad_elem) {
                        std::cout << elem << " ";
                    }
                }
                std::cout << std::endl;*/
            }
        }
        else if (parallelImplCpu == OpenMP) {
            for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                grad = (*it)->backwardCPUopenMP(grad, learningRate, backward_out_neurons_num_threads, backward_in_neurons_num_threads);
                //std::cout << "New elements in grad: " ;
                /*for (auto grad_elem : grad) {
                    for (auto elem : grad_elem) {
                        std::cout << elem << " ";
                    }
                }
                std::cout << std::endl;*/
            }
        }
    }
    else if (device == CUDA) {
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = (*it)->backwardCUDA(grad, learningRate);
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
    // float loss = - (label * std::log(output[0]) + (1 - label) * std::log(1 - output[0]));
    // return loss;
}

/*float NeuralNetwork::computeAccuracy(const std::vector<std::vector<float>>& inputs, const std::vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        correct += (predict(inputs[i]) == labels[i]);
    }
    return static_cast<float>(correct) / inputs.size();
}*/





