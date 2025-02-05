//
// Created by kevin on 08/12/24.
//

#include <chrono> // For timing
#include <iostream> // For input/output
#include <forward_cuda.h> // Header for CUDA forward pass
#include <cuda_runtime.h> // CUDA runtime API

// Define a function pointer type for activation functions
typedef float (*activationFunc)(float);

// ReLU activation function (device code)
__device__ float relu(float x) {
    return fmaxf(0.0f, x); // Returns max(0, x)
}

// Sigmoid activation function (device code)
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x)); // Returns 1 / (1 + exp(-x))
}

// Activation function selector (device code)
__device__ float activate(float x, ActivationFunctionType act_type) {
    if (act_type == RELU) {
        return relu(x); // Use ReLU activation
    } else if (act_type == SIGMOID) {
        return sigmoid(x); // Use Sigmoid activation
    }
    return x; // No activation (identity function)
}

// Shared memory declaration (used for tiling in matrix multiplication)
extern __shared__ float shared_memory[];

// CUDA kernel for matrix multiplication with activation function
__global__ void matMulKernel(float *a, float *b, float *ab, int N, int K, int M, ActivationFunctionType act_type, int TILE_WIDTH) {
    // Thread and block indices
    int tx = threadIdx.x, ty = threadIdx.y; // Thread IDs within the block
    int bx = blockIdx.x, by = blockIdx.y; // Block IDs within the grid

    // Shared memory pointers for tiles of matrices A and B
    float* s_a = (float*)shared_memory;
    float* s_b = s_a + TILE_WIDTH * TILE_WIDTH;

    // Global row and column indices for the output matrix
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float result = 0; // Accumulator for the dot product

    // Loop over tiles of the input matrices
    for (int p = 0; p < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        // Load elements of matrix A into shared memory
        s_a[ty * TILE_WIDTH + tx] = (row < N && p * TILE_WIDTH + tx < K) ? a[row * K + p * TILE_WIDTH + tx] : 0.0;
        // Load elements of matrix B into shared memory
        s_b[ty * TILE_WIDTH + tx] = (p * TILE_WIDTH + ty < K && col < M) ? b[(p * TILE_WIDTH + ty) * M + col] : 0.0;
        __syncthreads(); // Synchronize threads to ensure shared memory is loaded

        // Compute the dot product for the current tile
        for (int k = 0; k < TILE_WIDTH; ++k)
            result += s_a[ty * TILE_WIDTH + k] * s_b[k * TILE_WIDTH + tx];
        __syncthreads(); // Synchronize threads before loading the next tile
    }

    // Write the result to the output matrix with activation function applied
    if (row < N && col < M) {
        ab[row * M + col] = activate(ab[row * M + col] + result, act_type);
    }
}

// Host function to perform matrix multiplication on the GPU
void forwardMatMul(float *a, float *b, float *ab, int M, int K, int N, ActivationFunctionType act_type, int TILE_WIDTH) {
    float *d_a = nullptr, *d_b = nullptr, *d_ab = nullptr; // Device pointers
    size_t sizeA = N * K * sizeof(float); // Size of matrix A
    size_t sizeB = K * M * sizeof(float); // Size of matrix B
    size_t sizeAB = N * M * sizeof(float); // Size of output matrix AB

    // Allocate device memory for matrices A, B, and AB
    CHECK_CUDA_ERROR_F(cudaMalloc(&d_a, sizeA), {});
    CHECK_CUDA_ERROR_F(cudaMalloc(&d_b, sizeB), cudaFree(d_a));
    CHECK_CUDA_ERROR_F(cudaMalloc(&d_ab, sizeAB), { cudaFree(d_a); cudaFree(d_b); });

    // Copy matrices A, B, and AB from host to device
    CHECK_CUDA_ERROR_F(cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });
    CHECK_CUDA_ERROR_F(cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });
    CHECK_CUDA_ERROR_F(cudaMemcpy(d_ab, ab, sizeAB, cudaMemcpyHostToDevice), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });

    // Define block and grid dimensions for the kernel
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH); // Threads per block
    dim3 gridDim((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH); // Blocks per grid

    // Calculate shared memory size for tiles of matrices A and B
    size_t shared_mem_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

    // Measure kernel execution time
    auto start_kernel = std::chrono::high_resolution_clock::now();
    // Launch the matrix multiplication kernel
    matMulKernel<<<gridDim, blockDim, shared_mem_size>>>(d_a, d_b, d_ab, N, K, M, act_type, TILE_WIDTH);
    auto end_kernel = std::chrono::high_resolution_clock::now();
    elapsed_f_kernel += end_kernel - start_kernel; // Accumulate kernel execution time
    CHECK_CUDA_ERROR_F(cudaPeekAtLastError(), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });

    // Synchronize to check for kernel errors
    CHECK_CUDA_ERROR_F(cudaDeviceSynchronize(), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });

    // Copy the result matrix AB from device to host
    CHECK_CUDA_ERROR_F(cudaMemcpy(ab, d_ab, sizeAB, cudaMemcpyDeviceToHost), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });

    // Free device memory
    CHECK_CUDA_ERROR_F(cudaFree(d_a), {});
    CHECK_CUDA_ERROR_F(cudaFree(d_b), {});
    CHECK_CUDA_ERROR_F(cudaFree(d_ab), {});
}
