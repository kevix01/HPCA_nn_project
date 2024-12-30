//
// Created by kevin on 08/12/24.
//

#include <chrono>
#include <iostream>
#include <forward_cuda.h>
#include <cuda_runtime.h>

typedef float (*activationFunc)(float);

__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float activate(float x, ActivationFunctionType act_type) {
    if (act_type == RELU) {
        return relu(x);
    } else if (act_type == SIGMOID) {
        return sigmoid(x);
    }
    return x;
}

extern __shared__ float shared_memory[];

__global__ void matMulKernel(float *a, float *b, float *ab, int N, int K, int M, ActivationFunctionType act_type, int TILE_WIDTH) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    float* s_a = (float*)shared_memory;
    float* s_b = s_a + TILE_WIDTH * TILE_WIDTH;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float result = 0;

    for (int p = 0; p < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        s_a[ty * TILE_WIDTH + tx] = (row < N && p * TILE_WIDTH + tx < K) ? a[row * K + p * TILE_WIDTH + tx] : 0.0;
        s_b[ty * TILE_WIDTH + tx] = (p * TILE_WIDTH + ty < K && col < M) ? b[(p * TILE_WIDTH + ty) * M + col] : 0.0;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            result += s_a[ty * TILE_WIDTH + k] * s_b[k * TILE_WIDTH + tx];
        __syncthreads();
    }

    if (row < N && col < M) {
        ab[row * M + col] = activate(ab[row * M + col] + result, act_type);
    }
}

/*void forwardMatMul(float *a, float *b, float *ab, int M, int K, int N, ActivationFunctionType act_type, int TILE_WIDTH) {
    float *d_a, *d_b, *d_ab;
    size_t sizeA = N * K * sizeof(float);
    size_t sizeB = K * M * sizeof(float);
    size_t sizeAB = N * M * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_a, sizeA);
    cudaMalloc(&d_b, sizeB);
    cudaMalloc(&d_ab, sizeAB);

    // Copy data from host to device
    cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ab, ab, sizeAB, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Calculate shared memory size
    size_t shared_mem_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

    // Launch the kernel
    matMulKernel<<<gridDim, blockDim, shared_mem_size>>>(d_a, d_b, d_ab, N, K, M, act_type, TILE_WIDTH);

    // Copy results from device to host
    cudaMemcpy(ab, d_ab, sizeAB, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_ab);
}*/

void forwardMatMul(float *a, float *b, float *ab, int M, int K, int N, ActivationFunctionType act_type, int TILE_WIDTH) {
    float *d_a = nullptr, *d_b = nullptr, *d_ab = nullptr;
    size_t sizeA = N * K * sizeof(float);
    size_t sizeB = K * M * sizeof(float);
    size_t sizeAB = N * M * sizeof(float);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, sizeA), {});
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, sizeB), cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ab, sizeAB), { cudaFree(d_a); cudaFree(d_b); });

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, sizeA, cudaMemcpyHostToDevice), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, b, sizeB, cudaMemcpyHostToDevice), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });
    CHECK_CUDA_ERROR(cudaMemcpy(d_ab, ab, sizeAB, cudaMemcpyHostToDevice), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });

    // Define block and grid dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Calculate shared memory size
    size_t shared_mem_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);

    auto start_kernel = std::chrono::high_resolution_clock::now();
    // Launch the kernel
    matMulKernel<<<gridDim, blockDim, shared_mem_size>>>(d_a, d_b, d_ab, N, K, M, act_type, TILE_WIDTH);
    auto end_kernel = std::chrono::high_resolution_clock::now();
    elapsed_f_kernel += end_kernel - start_kernel;
    CHECK_CUDA_ERROR(cudaPeekAtLastError(), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });

    // Synchronize to check for errors
    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });

    // Copy results from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(ab, d_ab, sizeAB, cudaMemcpyDeviceToHost), { cudaFree(d_a); cudaFree(d_b); cudaFree(d_ab); });

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_a), {});
    CHECK_CUDA_ERROR(cudaFree(d_b), {});
    CHECK_CUDA_ERROR(cudaFree(d_ab), {});
}


