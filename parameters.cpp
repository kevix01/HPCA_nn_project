//
// Created by kevin on 18/12/24.
//

#include "parameters.h"
#include <cstdlib> // For std::atoi and std::atof
#include <iostream>
#include <vector>

// Constructor for Parameters
Parameters::Parameters(int argc, char* argv[]) {
    parseArguments(argc, argv); // Parse command-line arguments
}

// Getter for the CPU parallelism implementation
ParallelImplCpu Parameters::getParallelImplCpu() const {
    return parallel_impl_cpu;
}

// Getter for the number of OpenMP threads
int Parameters::getOpenmpThreads() const {
    return openmp_threads;
}

// Getter for the CUDA forward kernel tile size
int Parameters::getCudaFTileSize() const {
    return cuda_f_tile_size;
}

// Getter for the CUDA backward kernel block size
int Parameters::getCudaBBlockSize() const {
    return cuda_b_block_size;
}

// Getter for the device type (CPU or CUDA)
DeviceType Parameters::getDevice() const {
    return device;
}

// Getter for the training batch size
int Parameters::getTrainBatchSize() const {
    return train_batch_size;
}

// Getter for the prediction batch size
int Parameters::getPredictBatchSize() const {
    return predict_batch_size;
}

// Getter for the number of training epochs
int Parameters::getTrainEpochs() const {
    return train_epochs;
}

// Getter for the learning rate
float Parameters::getLearningRate() const {
    return learning_rate;
}

// Getter for the number of neurons in the first hidden layer
int Parameters::getNeuronsFirstHiddenLayer() const {
    return neurons_first_hidden_layer;
}

// Getter for the number of neurons in the second hidden layer
int Parameters::getNeuronsSecondHiddenLayer() const {
    return neurons_second_hidden_layer;
}

// Getter for the seed for weights initialization
int Parameters::getWeightsInitSeed() const {
    return weights_init_seed;
}

// Getter for the seed for shuffling the training data
int Parameters::getTraindataShuffleSeed() const {
    return traindata_shuffle_seed;
}

// Parse command-line arguments
void Parameters::parseArguments(int argc, char* argv[]) {
    // List of valid argument prefixes
    const std::vector<std::string> validPrefixes = {
        "--openmp-threads=",
        "--parallel-impl-cpu=",
        "--cuda-f-tile-size=",
        "--cuda-b-block-size=",
        "--device=",
        "--train-batch-size=",
        "--predict-batch-size=",
        "--train-epochs=",
        "--learning-rate=",
        "--first-hlayer-neurons=",
        "--second-hlayer-neurons=",
        "--weights-init-seed=",
        "--traindata-shuffle-seed=",
        "--help"
    };

    // Iterate through all command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i]; // Convert argument to a string

        // Check if the argument starts with a valid prefix
        bool isValidArgument = false;
        for (const auto& prefix : validPrefixes) {
            if (arg.find(prefix) == 0) {
                isValidArgument = true;
                break;
            }
        }

        // If the argument is not valid, print an error and exit
        if (!isValidArgument) {
            std::cerr << "Error: Unknown argument '" << arg << "'.\n";
            std::cerr << "Use --help for a list of valid arguments.\n";
            exit(1);
        }

        // Check for OpenMP threads argument
        if (arg.find("--openmp-threads=") == 0) {
            int value = std::atoi(arg.substr(17).c_str());
            if (value > 0) {
                openmp_threads = value;
            } else {
                std::cerr << "Error: --openmp-threads must be a positive integer.\n";
                exit(1);
            }
        }
        // Check for CPU parallelism implementation argument
        else if (arg.find("--parallel-impl-cpu=") == 0) {
            std::string parallel_impl_cpu_text = arg.substr(20);
            if (parallel_impl_cpu_text == "openmp") {
                parallel_impl_cpu = OpenMP;
            } else if (parallel_impl_cpu_text == "no") {
                parallel_impl_cpu = No;
            } else {
                std::cerr << "Error: --parallel-impl-cpu must be 'openmp' or 'no'.\n";
                exit(1);
            }
        }
        // Check for CUDA forward kernel tile size argument
        else if (arg.find("--cuda-f-tile-size=") == 0) {
            int value = std::atoi(arg.substr(19).c_str());
            if (value > 0) {
                cuda_f_tile_size = value;
            } else {
                std::cerr << "Error: --cuda-f-tile-size must be a positive integer.\n";
                exit(1);
            }
        }
        // Check for CUDA backward kernel block size argument
        else if (arg.find("--cuda-b-block-size=") == 0) {
            int value = std::atoi(arg.substr(20).c_str());
            if (value > 0) {
                cuda_b_block_size = value;
            } else {
                std::cerr << "Error: --cuda-b-block-size must be a positive integer.\n";
                exit(1);
            }
        }
        // Check for device type argument
        else if (arg.find("--device=") == 0) {
            std::string device_text = arg.substr(9);
            if (device_text == "cuda") {
                device = CUDA;
            } else if (device_text == "cpu") {
                device = CPU;
            } else {
                std::cerr << "Error: --device must be 'cpu' or 'cuda'.\n";
                exit(1);
            }
        }
        // Check for training batch size argument
        else if (arg.find("--train-batch-size=") == 0) {
            int value = std::atoi(arg.substr(19).c_str());
            if (value > 0) {
                train_batch_size = value;
            } else {
                std::cerr << "Error: --train-batch-size must be a positive integer.\n";
                exit(1);
            }
        }
        // Check for prediction batch size argument
        else if (arg.find("--predict-batch-size=") == 0) {
            int value = std::atoi(arg.substr(21).c_str());
            if (value > 0) {
                predict_batch_size = value;
            } else {
                std::cerr << "Error: --predict-batch-size must be a positive integer.\n";
                exit(1);
            }
        }
        // Check for training epochs argument
        else if (arg.find("--train-epochs=") == 0) {
            int value = std::atoi(arg.substr(15).c_str());
            if (value > 0) {
                train_epochs = value;
            } else {
                std::cerr << "Error: --train-epochs must be a positive integer.\n";
                exit(1);
            }
        }
        // Check for learning rate argument
        else if (arg.find("--learning-rate=") == 0) {
            float value = std::atof(arg.substr(16).c_str());
            if (value > 0.0f) {
                learning_rate = value;
            } else {
                std::cerr << "Error: --learning-rate must be a positive float.\n";
                exit(1);
            }
        }
        // Check for number of neurons in the first hidden layer argument
        else if (arg.find("--first-hlayer-neurons=") == 0) {
            int value = std::atoi(arg.substr(23).c_str());
            if (value > 0) {
                neurons_first_hidden_layer = value;
            } else {
                std::cerr << "Error: --first-hlayer-neurons must be a positive integer.\n";
                exit(1);
            }
        }
        // Check for number of neurons in the second hidden layer argument
        else if (arg.find("--second-hlayer-neurons=") == 0) {
            int value = std::atoi(arg.substr(24).c_str());
            if (value >= 0) { // Allow 0 to indicate no second hidden layer
                neurons_second_hidden_layer = value;
            } else {
                std::cerr << "Error: --second-hlayer-neurons must be a non-negative integer.\n";
                exit(1);
            }
        }
        // Check for weights initialization seed argument
        else if (arg.find("--weights-init-seed=") == 0) {
            int value = std::atoi(arg.substr(20).c_str());
            if (value >= 0) { // Allow 0 or any positive integer
                weights_init_seed = value;
            } else {
                std::cerr << "Error: --weights-init-seed must be a non-negative integer.\n";
                exit(1);
            }
        }
        // Check for training data shuffle seed argument
        else if (arg.find("--traindata-shuffle-seed=") == 0) {
            int value = std::atoi(arg.substr(25).c_str());
            if (value >= 0) { // Allow 0 or any positive integer
                traindata_shuffle_seed = value;
            } else {
                std::cerr << "Error: --traindata-shuffle-seed must be a non-negative integer.\n";
                exit(1);
            }
        }
        // Check for help argument
        else if (arg.find("--help") == 0) {
            // Print help message with descriptions of all parameters
            std::cout << "Usage: ./hpca_nn [OPTIONS]\n";
            std::cout << "Options:\n";
            std::cout << "* Device parameters:\n";
            std::cout << "--device=DEVICE\t\t\t\tDevice to use ('cpu' or 'cuda', mandatory)\n";
            std::cout << "--parallel-impl-cpu=IMPL\t\tParallel implementation for CPU ('openmp' or 'no', default: 'no')\n";
            std::cout << "--openmp-threads=NUM\t\t\tNumber of threads for OpenMP parallelism (default: 1)\n";
            std::cout << "--cuda-f-tile-size=NUM\t\t\tTile size for CUDA forward kernel (default: 16)\n";
            std::cout << "--cuda-b-block-size=NUM\t\t\tBlock size for CUDA backward kernel (default: 256)\n";
            std::cout << "* Model architecture:\n";
            std::cout << "--first-hlayer-neurons=NUM\t\tNumber of neurons in the first hidden layer (default: 50)\n";
            std::cout << "--second-hlayer-neurons=NUM\t\tNumber of neurons in the second hidden layer (default: 10)\n";
            std::cout << "* Model parameters:\n";
            std::cout << "--train-batch-size=NUM\t\t\tSize of the training batch (default: 50)\n";
            std::cout << "--predict-batch-size=NUM\t\tSize of the prediction batch (default: 50)\n";
            std::cout << "--train-epochs=NUM\t\t\tNumber of training epochs (default: 20)\n";
            std::cout << "--learning-rate=NUM\t\t\tLearning rate (default: 0.01)\n";
            std::cout << "--weights-init-seed=NUM\t\t\tSeed for weights initialization (default: 0)\n";
            std::cout << "--traindata-shuffle-seed=NUM\t\tSeed for shuffling the training data (default: 0)\n";
            exit(0); // Exit the program after printing the help message
        }
    }

    // Additional validation checks
    if (device == NONE) {
        std::cerr << "Error: --device must be specified ('cpu' or 'cuda').\n";
        exit(1);
    }
    if (parallel_impl_cpu == OpenMP && openmp_threads <= 0) {
        std::cerr << "Error: --openmp-threads must be a positive integer when using OpenMP.\n";
        exit(1);
    }
}
