# HPCA Neural Network Project

This project implements a neural network with support for both CPU and CUDA execution. It includes features like forward and backward propagation, activation functions (ReLU and Sigmoid), and parallelization using OpenMP for CPU and CUDA for GPU.

## Table of Contents
1. Project Overview
2. Dependencies
3. Build Instructions
4. Execution
5. Command-Line Arguments
6. Project Structure
7. License

---
## Project Overview

The project consists of a neural network implementation that can be executed on either the CPU or GPU (CUDA). It supports:
- Forward and backward propagation.
- Activation functions: ReLU and Sigmoid.
- Parallelization: OpenMP for CPU and CUDA for GPU.
- Command-line configuration: Device type, batch size, learning rate, and more.

The program automatically trains the network on the REJAFADA dataset, which contains **6824 features** and **1996 samples**. During training, the loss for each epoch is displayed, allowing you to monitor the model's progress. After training, the program performs a prediction step and calculates the final accuracy of the model.

### Key Features:
- **Training**: The network is trained using mini-batch gradient descent. The loss for each epoch is shown.
- **Prediction**: After training, the network predicts the labels for the dataset and computes the accuracy.
- **Flexibility**: The network architecture and training parameters can be configured via command-line arguments, such as the number of neurons in hidden layers, batch size, and learning rate.
- **Performance**: Forward and backward can be parallelized using OpenMP for CPU and CUDA for GPU.

### Dataset:
- **REJAFADA**: A dataset with 6824 features and 1996 samples. The dataset is loaded and normalized.
---

## Dependencies

To build and run this project, you need the following dependencies:
- CMake (version 3.28 or higher)
- CUDA Toolkit (if using GPU)
- OpenMP (if using CPU parallelism)
- C++ Compiler with C++20 support (e.g., g++ or clang++)

---

## Build Instructions

1. Clone the repository:  
   `git clone https://github.com/your-username/hpca_nn.git`  
   `cd hpca_nn`

2. Create a build directory:  
   `mkdir build`  
   `cd build`

3. Run CMake:  
   `cmake ..`  
   This will generate the necessary build files.

4. Compile the project:  
   `make`  
   This will compile the project and generate the executable `hpca_nn`.

---

## Execution

After building the project, you can run the executable with the following command:
./hpca_nn [OPTIONS]

### Example Commands

1. Run on CPU with OpenMP:  
   `./hpca_nn --device=cpu --parallel-impl-cpu=openmp --openmp-threads=4`

2. Run on CUDA:  
   `./hpca_nn --device=cuda --cuda-f-tile-size=32 --cuda-b-block-size=1024`

3. Set training parameters:  
   `./hpca_nn --device=cpu --train-batch-size=30 --train-epochs=15 --learning-rate=0.02`

4. Set a different neural network configuration:  
   `./hpca_nn --device=cpu --first-hlayer-neurons=25 --second-hlayer-neurons=5`

5. Use only the first hidden layer:  
   `./hpca_nn --device=cpu --second-hlayer-neurons=0`

---

## Command-Line Arguments

The following command-line arguments are supported:

| Argument                        | Description                                                                 | Default Value |
|---------------------------------|-----------------------------------------------------------------------------|---------------|
| --device=DEVICE               | Device to use (cpu or cuda).                                            | cpu         |
| --parallel-impl-cpu=IMPL      | Parallel implementation for CPU (openmp or no).                        | no          |
| --openmp-threads=NUM          | Number of threads for OpenMP parallelism.                                   | 1           |
| --cuda-f-tile-size=NUM        | Tile size for CUDA forward kernel.                                          | 16          |
| --cuda-b-block-size=NUM       | Block size for CUDA backward kernel.                                        | 256         |
| --train-batch-size=NUM        | Size of the training batch.                                                 | 50          |
| --predict-batch-size=NUM      | Size of the prediction batch.                                               | 50          |
| --train-epochs=NUM            | Number of training epochs.                                                  | 20          |
| --learning-rate=NUM           | Learning rate for training.                                                 | 0.01        |
| --first-hlayer-neurons=NUM    | Number of neurons in the first hidden layer.                                | 50          |
| --second-hlayer-neurons=NUM   | Number of neurons in the second hidden layer (use 0 to disable).          | 10          |
| --weights-init-seed=NUM       | Seed for weights initialization.                                            | 0           |
| --traindata-shuffle-seed=NUM  | Seed for shuffling the training data.                                       | 0           |
| --help                        | Print help message and exit.                                                | N/A           |

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

For any questions or issues, please open an issue on the GitHub repository.
